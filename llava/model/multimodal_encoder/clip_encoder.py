import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTokenizer, CLIPTextModel, CLIPModel
import math
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def normalize_visiontrim_method(method):
    if isinstance(method, str) and method.lower() == "visiontrim":
        return "VisionTrim"
    return method

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1] # top-k 数量
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim): # 升维与idx对齐
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0) # 将idx位置的值设为0，其他位置保持原值
    compl, _ = torch.sort(masked, dim=-1, descending=False) # 排序后，idx位置的值为0，其他位置为原值，且idx位置在前面
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,))) # 去掉前n_idx个值（即idx位置的值），剩下的就是补集索引
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

# LTAM algorithm 局部亲和力模块
def build_token_affinity(tokens, window_size=3, w1=1.0, w2=1.0, w3=0.5):
    """
    Build token affinity kernel and calculate importance (supports batch processing)
    Args:
        tokens: Input token features [bs, 576, 1024]
        window_size: Local window size
        w1, w2: Feature and position smoothing parameters
        w3: Position term weight
    Returns:
        local_token_importance: [bs, 576]
    """
    bs, B, C = tokens.shape  # bs=batch_size, B=576, C=1024
    H = W = int(math.sqrt(B))  # H=W=24

    # Build position encoding P_ij (shared across all batches)
    y_coords = torch.arange(H, device=tokens.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=tokens.device, dtype=torch.float32)
    y_coords = y_coords.view(-1, 1).repeat(1, W).view(-1)
    x_coords = x_coords.repeat(H)
    positions = torch.stack([y_coords, x_coords], dim=1)  # [576, 2]

    # Prepare result storage for each batch
    affinity = torch.zeros(bs, B, window_size * window_size, device=tokens.device, dtype=torch.float32)

    pad_size = window_size // 2

    # Process each batch separately
    for batch_idx in range(bs):
        # Calculate standard deviation for current batch
        sigma_feat = tokens[batch_idx].var()
        sigma_pos = positions.var()

        for i in range(H):
            for j in range(W):
                curr_idx = i * W + j
                # F_ij: Current position features
                curr_feat = tokens[batch_idx, curr_idx]  # [1024] 当前要处理的token的特征
                # P_ij: Current position coordinates
                curr_pos = positions[curr_idx]  # [2]

                # Get local neighborhood 𝒩(i,j)
                local_indices = []
                for di in range(-pad_size, pad_size + 1):
                    for dj in range(-pad_size, pad_size + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            local_indices.append(ni * W + nj) # 获取一维index

                # Calculate κ_feat
                local_feats = tokens[batch_idx, local_indices]  # [n_neighbors, 1024]
                feat_diff = ((curr_feat.unsqueeze(0) - local_feats) ** 2).sum(dim=1) #计算当前token与邻居token的特征差异
                kappa_feat = -feat_diff / (w1 * sigma_feat) #缩放，特征相似度核，差异越小，相似度越高

                # Calculate κ_pos
                local_pos = positions[local_indices]  # [n_neighbors, 2]
                pos_diff = ((curr_pos.unsqueeze(0) - local_pos) ** 2).sum(dim=1) # 计算当前token与邻居token的位置信息差异
                kappa_pos = -pos_diff / (w2 * sigma_pos) #缩放，位置相似度核，差异越小，相似度越高

                # Calculate combined affinity kernel κ
                kappa = kappa_feat + w3 * kappa_pos # 权重融合

                # Fill results
                affinity[batch_idx, curr_idx, :len(local_indices)] = kappa

    # Calculate token importance I_ij, remove last dimension
    local_token_importance = affinity.mean(dim=2)  # [bs, 576]
    local_token_importance = torch.softmax(local_token_importance, dim=-1)

    return local_token_importance

# Adaptive Variance-based Weighting     方法：根据全局注意力和局部重要性的方差动态调整权重，方差较小的指标权重较大，因为方差小意味着该指标在不同token之间的分布更集中，更具有区分度。
def combine_importance(cls_attn, local_token_importance, method='weighted_sum', alpha=0.5):
    """
    Combine global attention and local importance.
    Args:
        cls_attn: Global attention scores [bs, 576]
        local_token_importance: Local importance scores [bs, 576]
        method: Combination method ['weighted_sum', 'geometric', 'max', 'adaptive']
        alpha: Weight of global importance (0~1)
    Returns:
        combined_importance: Combined importance scores [bs, 576]
    """
    # 1. Normalize both metrics to the same scale
    cls_attn_norm = torch.softmax(cls_attn, dim=-1)
    local_imp_norm = torch.softmax(local_token_importance, dim=-1)
    # 'weighted_sum' 'geometric' 'max' 'harmonic' 'adaptive'
    if method == 'weighted_sum':
        # Simple weighted sum
        return alpha * cls_attn_norm + (1 - alpha) * local_imp_norm
    elif method == 'geometric':
        # Geometric mean, can better balance the two metrics
        return torch.sqrt(cls_attn_norm * local_imp_norm)
    elif method == 'max':
        # Take the larger value of the two metrics, emphasizing the upper bound of importance
        return torch.maximum(cls_attn_norm, local_imp_norm)
    elif method == 'harmonic':
        # Harmonic mean: more sensitive to extreme values
        return 2 * (cls_attn_norm * local_imp_norm) / (cls_attn_norm + local_imp_norm + 1e-8)
    elif method == 'adaptive':
        # Adaptive weighting: dynamically adjust weights based on the variance of the two metrics
        cls_var = cls_attn_norm.var(dim=-1, keepdim=True)
        local_var = local_imp_norm.var(dim=-1, keepdim=True)
        # The smaller the variance, the more certain the metric is, and the greater the weight should be
        cls_weight = local_var / (cls_var + local_var)
        local_weight = cls_var / (cls_var + local_var)
        return cls_weight * cls_attn_norm + local_weight * local_imp_norm
    else:
        raise ValueError(f"Unknown method: {method}")

# normalize score to [0,1], avoid negative value and scale difference between different metrics when combining them
def normalize_score(score,eps=1e-6):
    score_min = score.min(dim=-1, keepdim=True).values
    score_max = score.max(dim=-1, keepdim=True).values
    normalized_score = (score - score_min) / (score_max - score_min + eps)
    return normalized_score

class UnionFind:
    def _init_(self, n):
        self.parent = list(range(n))

    def find(self, x):  #compress path and find root
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

    def groups(self):
        group_dict = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            group_dict.setdefault(root, []).append(i)
        return list(group_dict.values())



def build_dynamic_token_groups(
        image_features, #[N,C] per image
        position_ids,
        similarity_threshold=0.9,
        grid_size = 24,
        neighbor_window=1,
):
    N = image_features.shape[0]
    uf = UnionFind(N)

    features_norm = F.normalize(image_features.float(), p=2, dim=-1)
    offsets = torch.tensor([(i,j) for i in range(-neighbor_window,neighbor_window+1) for j in range(-neighbor_window,neighbor_window+1) if not (i==0 and j==0)],
                           device = image_features.device,
                           dtype = torch.long,
                           ) # [num_neighbors, 2]
    position_ids = position_ids.to(device=image_features.device, dtype=torch.long)

    x = (position_ids % grid_size).unsqueeze(1)
    y = (position_ids // grid_size).unsqueeze(1)
    coords = torch.cat([x,y], dim=-1) # [N,2]
    neighbor_coords = coords[:, None, :] + offsets[None, :, :] # [N, num_neighbors, 2]

    valid_mask = (neighbor_coords[:,:,0] >= 0) & (neighbor_coords[:,:,0] < grid_size)& (neighbor_coords[:,:,1] >= 0) & (neighbor_coords[:,:,1] < grid_size)

    # convert neighbor coordinates back to position ids
    neighbor_pos = neighbor_coords[:,:,0] + neighbor_coords[:,:,1] * grid_size # [N, num_neighbors]
    neighbor_pos = neighbor_pos.clamp(0, grid_size*grid_size-1)

    pos_to_local = torch.full(
        (grid_size*grid_size,),
        -1,
        device=image_features.device,
        dtype=torch.long
    )
    pos_to_local[position_ids] = torch.arange(N, device=image_features.device)

    neighbor_local_idx = pos_to_local[neighbor_pos] # [N, num_neighbors]

    src_idx = torch.arange(N, device=image_features.device)[:, None].expand_as(neighbor_local_idx) # [N, num_neighbors]

    edge_mask = (
        valid_mask
        &(neighbor_local_idx >=0)
        &(neighbor_local_idx > src_idx)
    )
    src_edges = src_idx[edge_mask]
    dst_edges = neighbor_local_idx[edge_mask]

    if src_edges.numel() == 0:
        return uf.groups()

    edge_similarity = (
        features_norm[src_edges] * features_norm[dst_edges]
    ).sum(dim=-1) # [num_edges]

    keep_edges = edge_similarity >= similarity_threshold

    src_edges = src_edges[keep_edges]
    dst_edges = dst_edges[keep_edges]

    for src, dst in zip(src_edges.tolist(), dst_edges.tolist()):
        uf.union(src, dst)

    return uf.groups()

    # for i in range(N):
    #     pos = int(position_ids[i].item())
    #     x = (pos % grid_size).unsqueeze(1)
    #     y = (pos // grid_size).unsqueeze(1)
    #     for di, dj in offsets:
    #         ni, nj = x + di,y + dj
    #         if ni < 0 or ni >= grid_size or nj < 0 or nj >= grid_size:
    #             continue
    #         neighbor_pos = ni + nj * grid_size
    #         j = pos_to_local.get(neighbor_pos, None)
    #         if j is None or j <= i:
    #             continue
    #         sim = torch.matmul(features_norm[i], features_norm[j])
    #         if sim >= similarity_threshold:
    #             uf.union(i, j)




def groupwise_merge_similar_tokens(
    image_features,
    metric=None,
    cls_score=None,
    position_ids=None,
    max_merge_num=64,
    similarity_threshold=0.90,
    min_group_size=8,
    alpha=0.3,  # CLS score
    beta=0.4,   # spatial score
    gamma=0.3,  # metric score
):
    B, N, C = image_features.shape
    # compute cosine similarity
    image_features_norm = F.normalize(image_features.float(),p = 2, dim=-1)
    cosineSimilarity = image_features_norm@image_features_norm.transpose(-2, -1) # cosineSimilarity.shape = [B,N,N]

    #mask self cosine similarity
    diagMask = torch.eye(cosineSimilarity.size(-1),dtype = torch.bool, device=cosineSimilarity.device)
    cosineSimilarity = cosineSimilarity.masked_fill(diagMask.unsqueeze(0), float('-inf'))


    nearest_similarity = cosineSimilarity.max(dim=-1)[0] # nearest_similarity.shape = [B, N]
    candidate = None

    if similarity_threshold is not None:
        nearest_similarity = nearest_similarity.clamp_min(similarity_threshold)


    merged_image_features = None
    merged_metric = None
    merged_position_ids = None
    merge_pairs = None

    return merged_image_features, merged_metric, merged_position_ids, merge_pairs

def aggregate_rest_tokens_into_selected(selected_image_features, other_image_features, rest_token_scores=None, vtc_times=1):
    """Aggregate discarded visual tokens back into the selected visual tokens."""
    if selected_image_features.size(1) == 0 or other_image_features.size(1) == 0:
        return selected_image_features

    if selected_image_features.dim() != 3 or other_image_features.dim() != 3:
        raise ValueError("VisionTrim aggregation expects [batch, tokens, hidden] tensors.")
    if selected_image_features.size(0) != other_image_features.size(0):
        raise ValueError("Selected tokens and rest tokens must have the same batch size.")
    if selected_image_features.size(2) != other_image_features.size(2):
        raise ValueError("Selected tokens and rest tokens must have the same hidden size.")
    if rest_token_scores is not None and rest_token_scores.shape != other_image_features.shape[:2]:
        raise ValueError("rest_token_scores must have shape [batch, rest_tokens].")

    # 用 fp32 做相似度和加权平均，避免 fp16 下小权重/归一化带来的数值不稳定。
    compute_dtype = torch.float32
    target_hidden = selected_image_features.to(dtype=compute_dtype)
    hidden_to_merge = other_image_features.to(dtype=compute_dtype)

    if rest_token_scores is None:
        weights = torch.ones(
            other_image_features.shape[:2],
            dtype=compute_dtype,
            device=other_image_features.device,
        )
    else:
        weights = rest_token_scores.to(device=other_image_features.device, dtype=compute_dtype).clamp_min(0)

    merge_times = max(1, int(vtc_times or 1))
    for _ in range(merge_times):
        selected_norm = F.normalize(target_hidden, dim=-1)
        rest_norm = F.normalize(hidden_to_merge, dim=-1)
        similarity = torch.matmul(rest_norm, selected_norm.transpose(-2, -1))

        assign_one_hot = torch.zeros(
            hidden_to_merge.shape[0],
            hidden_to_merge.shape[1],
            target_hidden.shape[1],
            dtype=compute_dtype,
            device=hidden_to_merge.device,
        )
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

        weighted_assign = assign_one_hot * weights.unsqueeze(-1)
        counts = weighted_assign.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        aggregated_hidden = torch.matmul(weighted_assign.transpose(1, 2), hidden_to_merge) / counts
        target_hidden = target_hidden + aggregated_hidden

    return target_hidden.to(dtype=selected_image_features.dtype)

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer # -2 22
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')


        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False) # 冻结视觉塔的参数

        # 原实现会在加载视觉塔时同时加载完整 CLIPModel、CLIPTokenizer 和 CLIPTextModel：
        # self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
        # self.clip_model.to(self.vision_tower.device)
        # self.clip_model.requires_grad_(False)
        # self.text_tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name)
        # self.text_encoder = CLIPTextModel.from_pretrained(self.vision_tower_name)
        # self.text_encoder.to(self.vision_tower.device)
        # self.text_encoder.requires_grad_(False)
        # 修改原因：只有显式纯 DVTS 消融（TGVC_token_num=0）不会用到文本引导分支；
        # README 的标准 VisionTrim 配置会在 Python 端把总 token 数拆成 DVTS+TGVC。
        # 提前加载完整 CLIP 文本/投影模块会增加显存占用，并在当前 transformers/accelerate
        # 环境下触发 meta tensor/OOM 问题。因此这里改成懒加载，只在 TGVC_token_num>0 时加载。
        self.clip_model = None
        self.text_tokenizer = None
        self.text_encoder = None

        self.is_loaded = True

    def load_text_guidance_model(self):
        # 懒加载文本引导模块：仅 TGVC 路径需要 text/image projection 和 text encoder。
        if self.clip_model is not None and self.text_tokenizer is not None and self.text_encoder is not None:
            self.clip_model.to(device=self.device, dtype=self.dtype)
            self.text_encoder.to(device=self.device, dtype=self.dtype)
            return

        offline_values = {"1", "ON", "YES", "TRUE"}
        local_files_only = (
            os.environ.get("HF_HUB_OFFLINE", "").upper() in offline_values
            or os.environ.get("TRANSFORMERS_OFFLINE", "").upper() in offline_values
        )

        # 原实现:
        # self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
        # self.text_tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name)
        # self.text_encoder = CLIPTextModel.from_pretrained(self.vision_tower_name)
        # 修改原因：POPE 标准 64-token 会启用 TGVC 并加载 CLIP text guidance。
        # 当前服务器网络访问 HuggingFace 不稳定；在离线模式下必须显式 local_files_only，
        # 否则 transformers 会先发 HEAD 请求，触发 SSL retry。
        self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name, local_files_only=local_files_only)
        self.clip_model.to(device=self.device, dtype=self.dtype)
        self.clip_model.requires_grad_(False)

        self.text_tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name, local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel.from_pretrained(self.vision_tower_name, local_files_only=local_files_only)
        self.text_encoder.to(device=self.device, dtype=self.dtype)
        self.text_encoder.requires_grad_(False)
    # 从ViT的输出中选择特征，可以选择patch token、cls token或者两者结合的特征
    def feature_select(self, image_forward_outs, images=None, is_two_parts=False, get_last_layer_with_cls=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # image_features = image_forward_outs.hidden_states[-1]
        if self.select_feature == 'patch':
            cls_token = image_features[:, 0:1]
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if get_last_layer_with_cls:
            return image_features, image_forward_outs.last_hidden_state
        return image_features, cls_token

    # 获取指定token_num数量的token，最后拼接上一个汇聚的全局token返回
    def token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio = 1/8, token_num=-1):
        if_adaptive = True

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]
        if token_num == 64:
            compress_num = 13
            step_length = 33
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576
        elif token_num == 128:
            compress_num = 40
            step_length = 18
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576
        else:
            compress_num = 95
            step_length = 9
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576



        if if_adaptive:
            # step_length = int(1/reduction_ratio)
            arithmetic_sequence = torch.arange(0, 575, int(step_length/3)).to(device=self.device)
            original_tensor_1d = idx.flatten().to(device=self.device)
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
            idx = concatenated_tensor
        else:
            # this is for training
            step_length = int(1/reduction_ratio)
            new_idx = torch.zeros((idx.size(0), idx.size(1)*2), dtype=torch.long).to(device=self.device)
            for i in range(idx.size(0)):
                arithmetic_sequence = torch.arange(int(step_length/2), 575, int(step_length)).to(device=self.device)
                original_tensor_1d = idx[i].flatten().to(device=self.device)
                filtered_sequence = arithmetic_sequence
                # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
                concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
                new_idx[i] = concatenated_tensor
            idx = new_idx

            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C] # Key without CLS

            x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C] 获取被选中token的特征
            x_others_attn = torch.gather(cls_attn, dim=1, index=idx)   #获取被选中token的cls_attn分数
            Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C] 获取被选中token的Key特征
            compl = complement_idx(idx, N)  # [B, N-1-left_tokens]  # 获取未被选中token的索引，即补集索引
            non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C] 获取未被选中token的特征
            non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C)) #   [B, N-1-left_tokens, C] 获取未被选中token的Key特征
            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens] # 获取未被选中token的cls_attn分数

            Key_others_norm = F.normalize(Key_others, p=2, dim=-1) # 对被选中token的Key特征进行归一化
            non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)
            B, left_tokens, C = x_others.size()
            updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0) # [1, 1, C] 当前被选中token的Key特征

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  # [1, i, C] 被选中token之前的token的Key特征
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) # [1, left_tokens-1-i, C] 被选中token之后的token的Key特征

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)   # [1, i, C] 被选中token之前的token的特征
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)    #   [1, left_tokens-1-i, C] 被选中token之后的token的特征
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)    # [1, left_tokens-1, C] 其他被选中token和未被选中token的特征
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  # [1, i] 被选中token之前的token的cls_attn分数
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)   # [1, left_tokens-1-i] 被选中token之后的token的cls_attn分数
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  # [1, left_tokens-1] 其他被选中token和未被选中token的cls_attn分数

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1) # [1, left_tokens-1, C] 其他被选中token和未被选中token的Key特征
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2)) # [1, 1, C] x [1, C, left_tokens-1] -> [1, 1, left_tokens-1] 计算当前被选中token的Key特征与其他被选中token和未被选中token的Key特征的余弦相似度

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = x_others[b, i, :]  + weighted_avg
                updated_x_others[b, i, :] = updated_center

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features.to(torch.float16)

    # the Text-Guided Vision Complement (TGVC) module
    def text_guided_vision_complement(self, images, text_features, token_num=23, vtc_times=1):
        # Encode text and obtain text features
        text_inputs = self.text_tokenizer(original_qs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        text_features_projection = self.clip_model.text_projection(text_features)

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)

        _, idx = torch.topk(metric, token_num, dim=1, largest=True)  # bs, token_num, sorted=True
        select_idx = torch.sort(idx, dim=1)[0]  # bs, token_num

        inverse_mask = torch.zeros((batch_size, image_features.size(1)), dtype=torch.bool, device=image_features.device)
        # Process each batch separately
        for b in range(batch_size):
            inverse_mask[b].scatter_(0, select_idx[b], True)
        inverse_mask = ~inverse_mask  # (B, 576)

        selected_image_features = torch.stack([image_features[b, select_idx[b]] for b in range(batch_size)])  # (B, token_num, 1024)
        # print(f"the shape of selected_image_features is {selected_image_features.shape}")
        other_image_features = torch.stack([image_features[b, inverse_mask[b]] for b in range(batch_size)])  # (B, 576-token_num, 1024)

        all_image_features_projection = self.clip_model.visual_projection(image_features)  # 1024->768
        all_image_features_norm = F.normalize(all_image_features_projection, dim=-1)
        other_image_features_projection = self.clip_model.visual_projection(other_image_features)
        other_image_features_norm = F.normalize(other_image_features_projection, dim=-1)

        text_features_norm = F.normalize(text_features_projection, dim=-1)
        text_to_image_similarity = torch.matmul(text_features_norm, other_image_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()

        token_scores = text_to_image_similarity.mean(dim=1)  # (B, 576-token_num, text_token_num) -> (B, 576-token_num)

        complement_num = 0

        if complement_num > 0:
            _, idx = torch.topk(token_scores, complement_num, dim=1, largest=True)
            other_idx = torch.sort(idx, dim=1)[0]
            target_hidden = torch.stack([other_image_features[b, other_idx[b]] for b in range(other_image_features.size(0))])
            hidden_to_merge = torch.stack([other_image_features[b, ~torch.isin(
                torch.arange(other_image_features.size(1), device=other_image_features.device),
                other_idx[b]
            )] for b in range(other_image_features.size(0))])

            vtc_times = 3
            for iteration in range(vtc_times):
                target_hidden_projection = self.clip_model.visual_projection(target_hidden)
                hidden_to_merge_projection = self.clip_model.visual_projection(hidden_to_merge)
                current_target_norm = F.normalize(target_hidden_projection, dim=-1)
                current_image2text_similarity = torch.matmul(current_target_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                current_text2image_similarity = torch.matmul(text_features_norm, current_target_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                hidden_to_merge_norm = F.normalize(hidden_to_merge_projection, dim=-1)
                hidden_image2text_similarity = torch.matmul(hidden_to_merge_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                metric_1 = torch.matmul(hidden_image2text_similarity, current_text2image_similarity)
                metric_normalized = metric_1 / metric_1.norm(dim=-1, keepdim=True)
                assign_one_hot = torch.zeros(hidden_to_merge.shape[0], hidden_to_merge.shape[1], complement_num, dtype=other_image_features.dtype, device=metric_normalized.device)
                assign_one_hot.scatter_(2, metric_normalized.argmax(dim=2).unsqueeze(-1), 1)
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                aggregated_hidden = torch.matmul(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                target_hidden = target_hidden + aggregated_hidden

            complement_tokens = target_hidden
            final_image_features = torch.cat([selected_image_features, complement_tokens], dim=1).to(images.dtype)
        else:
            final_image_features = selected_image_features

        return final_image_features

    def token_prune_visiontrim(self, images, if_adaptive=True, reduction_ratio=1/8, csa=False, token_num=23, TGVC_token_num=0, start_layer=None, dataset_name=None, original_qs=None, vtc_times=1):
        token_num = int(token_num)
        TGVC_token_num = int(TGVC_token_num or 0)
        if token_num < 0 or TGVC_token_num < 0:
            raise ValueError("VisionTrim token counts must be non-negative.")

        # Ensure text encoder and tokenizer are loaded
        if TGVC_token_num > 0 and original_qs is None:
            raise RuntimeError("TGVC_token_num > 0 requires original_qs for text-guided complement.")
        if TGVC_token_num > 0:
            # 原实现假设 load_model() 已经预加载 text guidance 组件。
            # 现在改为按需加载，避免 TGVC_token_num=0 的普通评估额外占用显存。
            self.load_text_guidance_model()
            # Encode text and obtain text features
            text_inputs = self.text_tokenizer(original_qs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.text_encoder(**text_inputs).last_hidden_state
                text_features_projection = self.clip_model.text_projection(text_features) # 投影到与视觉特征相同的维度

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True) # 获取ViT每一层的hidden_states和attentions，hidden_states包含每一层的输出特征，attentions包含每一层的注意力权重

        # the Dominant Vision Token Selection (DVTS) module
        image_features, cls_token = self.feature_select(image_forward_outs)  # Output hidden_states of the second-to-last layer of ViT
        image_features = image_features.to(images.dtype)
        batch_size = image_features.size(0)
        if token_num + TGVC_token_num > image_features.size(1):
            raise ValueError(
                f"Requested {token_num + TGVC_token_num} visual tokens, but only {image_features.size(1)} patch tokens are available."
            )

        local_token_importance = build_token_affinity(image_features)
        last_layer_attn = image_forward_outs.attentions[-2]  # bs, heads, 577, 577
        last_layer_attn = last_layer_attn.mean(dim=1)  # bs, 577, 577

        # 1. Extract attention matrix between image tokens
        image_attn = last_layer_attn[:, 1:, 1:]  # [bs, 576, 576]
        # Create mask to exclude diagonal (self-attention)
        bs, n_tokens = image_attn.shape[0], image_attn.shape[1]
        mask = torch.eye(n_tokens, dtype=torch.bool, device=image_attn.device)
        masked_attn = image_attn.masked_fill(mask, 0) # [bs, 576, 576] 将对角线元素（自注意力）设为0，保留非对角线元素（不同token之间的注意力）元素为True则填充0
        # 2. Calculate average and total attention received by each image token
        avg_received_attn = masked_attn.sum(dim=1) / (n_tokens - 1)  # [bs, 576] 每个token对其他token的平均注意力分配，即每个token接收的平均注意力
        sum_received_attn = masked_attn.sum(dim=1)  # [bs, 576]

        # Calculate average and total attention allocated by each image token to other tokens (即每个token对其他token的平均和总的注意力分配)
        avg_sent_attn = masked_attn.sum(dim=2) / (n_tokens - 1)
        sum_sent_attn = masked_attn.sum(dim=2)
        cls_attn = last_layer_attn[:, 0, 1:]  # bs, 576
        # 原实现计算了 cls_attn_sum，但后面 metric 仍然使用 cls_attn，导致 --layer/start_layer
        # 对视觉 token ranking 基本没有生效：
        # cls_attn_multi_layers = [attn.mean(dim=1)[:, 0, 1:] for attn in image_forward_outs.attentions[start_layer:-1]]
        # cls_attn_sum = torch.sum(torch.stack(cls_attn_multi_layers), dim=0)
        # 修改原因：论文里的 layer ablation 应该改变视觉 token 选择依据，而不是只改变 LLM
        # attention backend。这里优先使用 start_layer 到倒数第二层的 CLS attention 累积值；
        # 如果 start_layer 缺失或切片为空，则回退到原来的倒数第二层 cls_attn。
        cls_attn_for_metric = cls_attn
        if start_layer is not None:
            start_layer = int(start_layer)
            cls_attn_multi_layers = [attn.mean(dim=1)[:, 0, 1:] for attn in image_forward_outs.attentions[start_layer:-1]]
            if len(cls_attn_multi_layers) > 0:
                cls_attn_for_metric = torch.sum(torch.stack(cls_attn_multi_layers), dim=0)  # can be modified to operate min or max

        # Normalize
        cls_attn_norm = F.normalize(cls_attn, dim=-1) # 每个token与cls token的注意力分数进行归一化，得到全局重要性指标
        avg_received_attn_norm = F.normalize(avg_received_attn, dim=-1) # 每个token接收的平均注意力分配进行归一化
        # 原实现:
        # metric = combine_importance(cls_attn, local_token_importance, method='adaptive', alpha=0.5)
        # 修改原因同上：让 start_layer 控制的多层 CLS attention 真正参与 token ranking。
        metric = combine_importance(cls_attn_for_metric, local_token_importance, method='adaptive', alpha=0.5)

        _, idx = torch.topk(metric, token_num, dim=1, largest=True)  # bs, token_num, sorted=True
        select_idx = torch.sort(idx, dim=1)[0]  # bs, token_num 获取每个batch中token_num个最重要的token的索引，首先使用topk函数获取每个batch中token_num个最大值的索引，然后使用sort函数对这些索引进行排序，得到最终的select_idx，形状为[bs, token_num]，表示每个batch中被选中的token的索引
        inverse_mask = torch.zeros((batch_size, image_features.size(1)), dtype=torch.bool, device=image_features.device)
        # Process each batch separately 标记被选中的token的索引位置为True，其他位置为False，得到一个布尔类型的mask，形状为[bs, 576]，其中被选中的token位置为True，未被选中的token位置为False
        for b in range(batch_size):
            inverse_mask[b].scatter_(0, select_idx[b], True)
        inverse_mask = ~inverse_mask  # (B, 576) 取反得到未被选中的token的mask，形状为[bs, 576]，其中未被选中的token位置为True，被选中的token位置为False
        selected_image_features = torch.stack([image_features[b, select_idx[b]] for b in range(batch_size)])  # (B, token_num, 1024)
        other_image_features = torch.stack([image_features[b, inverse_mask[b]] for b in range(batch_size)])  # (B, 576-token_num, 1024)
        other_token_scores = torch.stack([metric[b, inverse_mask[b]] for b in range(batch_size)])  # (B, 576-token_num)
        complement_num = TGVC_token_num

        if complement_num > 0:
            other_image_features_projection = self.clip_model.visual_projection(other_image_features)
            other_image_features_norm = F.normalize(other_image_features_projection, dim=-1)
            text_features_norm = F.normalize(text_features_projection, dim=-1)
            text_to_image_similarity = torch.matmul(text_features_norm, other_image_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
            token_scores = text_to_image_similarity.mean(dim=1)
            _, idx = torch.topk(token_scores, complement_num, dim=1, largest=True)
            other_idx = torch.sort(idx, dim=1)[0]
            target_hidden = torch.stack([other_image_features[b, other_idx[b]] for b in range(other_image_features.size(0))])  #[B, complement_num, C]
            hidden_to_merge = torch.stack([other_image_features[b, ~torch.isin(
                torch.arange(other_image_features.size(1), device=other_image_features.device),
                other_idx[b]
            )] for b in range(other_image_features.size(0))]) # 剩余非 top-k token 中未被选作补充 token 的特征集合

            vtc_times = 3
            for iteration in range(vtc_times):
                target_hidden_projection = self.clip_model.visual_projection(target_hidden)
                hidden_to_merge_projection = self.clip_model.visual_projection(hidden_to_merge)
                current_target_norm = F.normalize(target_hidden_projection, dim=-1)
                current_image2text_similarity = torch.matmul(current_target_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                current_text2image_similarity = torch.matmul(text_features_norm, current_target_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                hidden_to_merge_norm = F.normalize(hidden_to_merge_projection, dim=-1)
                hidden_image2text_similarity = torch.matmul(hidden_to_merge_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                metric_1 = torch.matmul(hidden_image2text_similarity, current_text2image_similarity)
                metric_normalized = metric_1 / metric_1.norm(dim=-1, keepdim=True) # L2归一化
                assign_one_hot = torch.zeros(hidden_to_merge.shape[0], hidden_to_merge.shape[1], complement_num, dtype=other_image_features.dtype, device=metric_normalized.device)
                assign_one_hot.scatter_(2, metric_normalized.argmax(dim=2).unsqueeze(-1), 1)  # 构建 one-hot 分配矩阵 [B, N_remaining, complement_num]
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1) # 每个补充 token 被多少剩余 token 分配
                aggregated_hidden = torch.matmul(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts   #可改进。。采用加权
                target_hidden = target_hidden + aggregated_hidden

            complement_tokens = target_hidden
            final_image_features = torch.cat([selected_image_features, complement_tokens], dim=1).to(images.dtype) # 最终压缩后的 token 特征集合
        else:
            # 原实现:
            # final_image_features = selected_image_features
            # 修改原因：论文里的 rest-token aggregation 应在 576 -> token_num 的视觉压缩阶段发生。
            # 当显式纯 DVTS 消融设置 TGVC_token_num=0 时，旧代码会把未选中的 other_image_features 直接丢弃；
            # 这里按特征相似度把 rest tokens 分配给保留 token，并用 DVTS metric 作为权重做加权聚合。
            final_image_features = aggregate_rest_tokens_into_selected(
                selected_image_features,
                other_image_features,
                rest_token_scores=other_token_scores,
                vtc_times=vtc_times,
            ).to(images.dtype)

        return cls_attn, final_image_features

    @torch.no_grad()
    def forward(self, images, method="none", dataset_name="none", start_layer=None, get_last_layer_with_cls=False, token_num=18, TGVC_token_num=0, original_qs=None):
        method = normalize_visiontrim_method(method)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # 原实现：
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                # 修改原因：feature_select() 返回 (patch_features, cls_token)，tuple 不能直接 .to()。
                # method=none 或 list 输入路径需要显式取 patch_features，VisionTrim 主路径不受影响。
                image_feature, _ = self.feature_select(image_forward_out)
                image_feature = image_feature.to(image.dtype)
                image_features.append(image_feature)
        else:
            if method == "VisionTrim":
                cls_attn, image_features = self.token_prune_visiontrim(images, if_adaptive=True, reduction_ratio=1/8, token_num=token_num, TGVC_token_num=TGVC_token_num, start_layer=start_layer, dataset_name=dataset_name, original_qs=original_qs)
                return cls_attn, image_features
            elif method == "llava_prumerge":
                image_features = self.token_prune_merge_advanced_plus(images, if_adaptive=token_num, reduction_ratio=1/8, token_num=token_num)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                # 原实现：
                # image_features = self.feature_select(images=images, image_forward_outs=image_forward_outs).to(images.dtype)
                # 修改原因同上：feature_select() 返回 tuple，这里只需要 patch token 特征。
                image_features, _ = self.feature_select(images=images, image_forward_outs=image_forward_outs)
                image_features = image_features.to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
