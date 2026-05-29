from abc import ABC, abstractmethod

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            if self.dtype:
                print(f"self.dtype is {self.dtype}")
                self.mm_projector = self.mm_projector.to(self.dtype)
            print(f"mm_projector has been build here.")
            print(f"the self.mm_projector is {self.mm_projector}")
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        print("Here use initialize_vision_modules")
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        # print(f"the pretrain_mm_mlp_adapter is {pretrain_mm_mlp_adapter}")
        mm_patch_merge_type = model_args.mm_patch_merge_type
        pretrain_abstractor = model_args.pretrain_abstractor

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            print(f"self.mm_projector is None")
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            print(f"self.mm_projector is not None")
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            # print(f"the pretrain_mm_projector_weights is {mm_projector_weights}")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            # print(f"the self.mm_projector is {self.mm_projector}")
            
            # 1. First completely recreate mm_projector
            self.mm_projector = nn.Sequential(
                nn.Linear(1024, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 4096, bias=True)
            ).to(self.dtype)  # Ensure correct data type

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            
            # Get pretrained weights
            processed_weights = get_w(mm_projector_weights, 'mm_projector')
            print(f"the shape of self.mm_projector is {self.mm_projector}")

            # 1. First completely recreate mm_projector
            self.mm_projector = nn.Sequential(
                nn.Linear(1024, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 4096, bias=True)
            ).to(self.dtype)  # Ensure correct data type

            # 2. Print shapes before initialization
            print("\nParameter shapes before initialization:")
            for name, param in self.mm_projector.named_parameters():
                print(f"{name}: {param.shape}")

            # 3. Manually initialize weights
            for name, param in self.mm_projector.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            # 4. Check if parameters are initialized
            for name, param in self.mm_projector.named_parameters():
                if param.numel() == 0:  # If parameter size is 0
                    print(f"Parameter {name} not initialized!")
       
            print("\nPretrained weights shapes:")
            for name, weight in processed_weights.items():
                print(f"processed_weights['{name}'].shape = {weight.shape}")

            print("\nCurrent model parameter shapes:")
            for name, param in self.mm_projector.named_parameters():
                print(f"self.mm_projector['{name}'].shape = {param.shape}")

            # Manually load weights and print
            for name, param in self.mm_projector.named_parameters():
                if name in processed_weights:
                    print(f"\nLoading {name}:")
                    print(f"Source shape: {processed_weights[name].shape}")
                    print(f"Target shape: {param.shape}")
                    # Ensure dimensions match
                    assert param.shape == processed_weights[name].shape, \
                        f"Shape mismatch for {name}: {param.shape} vs {processed_weights[name].shape}"
                    # Load weights
                    param.data.copy_(processed_weights[name])


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def hook_attn(module, input, output):
    outputs['desired_attn'] = output
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
class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images, original_qs=None):
        dataset_name = self.get_model().dataset_name if hasattr(self.get_model(), 'dataset_name') else None
        print(f"the original_qs in encode images is {original_qs}")
        if self.get_model().method == "VisionTrim":
            token_num = self.get_model().token_num
            cls_attn, image_features = self.get_model().get_vision_tower()(images, method=self.get_model().method, dataset_name=dataset_name, start_layer=self.get_model().layer, token_num=token_num, original_qs=original_qs)
        else:
            print(f"Here is the encode_images function in else")
            image_features = self.get_model().get_vision_tower()(images, method=self.get_model().method, dataset_name=dataset_name)
        if self.get_model().mm_projector[0].weight.device != image_features.device:
            self.get_model().mm_projector = self.get_model().mm_projector.to(image_features.device)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(  # 
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ): 
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1: #
            return input_ids, position_ids, attention_mask, past_key_values, None, labels,None

        if type(images) is list or images.ndim == 5: # 如果输入的 images 是一个 list，或者是一个 5 维的 tensor（batch_size, num_images, C, H, W），就说明每个样本可能有不同数量的图像，需要分别编码后再拼接。否则就直接把 images 当成一个 batch 来编码。
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'): # 
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1: # 如果图像被编码成了多个 token 了（比如 CLS token + patch token），就把这些 token 进行空间排列和拼接
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # view：把一维 token 序列恢复成空间网格
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type: 
                            # 外层网格：H_g × W_g，每个外层图像块内部：h × w 个 patch token，每个 token 特征维度：C
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # image_feature.shape = [H_g, W_g, h, w, C] 
                            
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3) # 两次 flatten：拼接外层图像块和内层 patch 网格, flatten后：image_feature.shape = [C, H_g * h, W_g * w]
                            image_feature = unpad_image(image_feature, image_sizes[image_idx]) # unpad_image：去除 padding 区域。 输出：image_feature.shape = [C, H_real, W_real]
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1) #image_feature.shape = [ H_real* (W_real+1),C]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous() # image_feature.shape = [H_g, h, W_g, w, C]
                            image_feature = image_feature.flatten(0, 3) # flatten：把外层图像块和内层 patch 网格一起拼接成一个 token 序列，image_feature.shape = [H_g * h * W_g * w, C]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0) # 把 CLS token 和 patch token 拼接在一起，image_feature.shape = [(H_g * h * W_g * w + 1), C]
                    else: # 如果图像被编码成了一个 token 了（比如 CLS token），就直接用这个 token 作为图像特征
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0) # self.model.image_newline[None]：给单 token 图像特征拼接一个 learnable 的 newline token，image_feature.shape = [C+1]
                    new_image_features.append(image_feature) 
                image_features = new_image_features 
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else: # 如果输入的 images 是一个 4 维的 tensor（batch_size, C, H, W），就说明每个样本只有一张图像，可以直接把这个 batch 的图像一起编码。
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False): 
            raise NotImplementedError 
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask # attention_mask [batch_size, seq_len], bool or float
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) # input_ids.shape[1] 是 seq_len
        if labels is None: # 如果没有提供 labels，就创建一个全是 IGNORE_INDEX 的 labels，这样在计算 loss 的时候就不会对这些位置计算 loss。
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids # 保留原始的 input_ids 以便后续使用
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)] # 根据 attention_mask 把 input_ids 中的 padding token 去掉，得到一个 list，每个元素是一个样本去掉 padding 后的 input_ids。
        #注意这里假设了 padding token 的位置在 attention_mask 中是 False。
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        images_idx = [torch.where(cur_input_ids == IMAGE_TOKEN_INDEX) for cur_input_ids in _input_ids] # 找到每个样本中 IMAGE_TOKEN_INDEX 的位置，这些位置就是图像 token 在 input_ids 中的位置。
        #images_idx 是一个 list，每个元素是一个 tuple，包含了该样本中所有图像 token 的位置索引。 IMAGE_TOKEN_INDEX 是一个特殊的 token id，表示图像 token 的位置。通过这个索引，我们可以知道图像 token 在 input_ids 中应该被替换成对应的 image_features。

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): #input_ids[batch_idx] 是一个样本去掉 padding 后的 input_ids，cur_input_ids.shape = [seq_len_no_pad]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 计算当前样本中图像 token 的数量，也就是需要插入多少个 image_features。
            if num_images == 0: # 如果当前样本中没有图像 token，就直接把文本 token 的 input_ids 转换成 input_embeds，然后添加到 new_input_embeds 中，同时把对应的 labels 添加到 new_labels 中。
                cur_image_features = image_features[cur_image_idx] #image_features[cur_image_idx].shape = [N_img_tokens, C]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids) #文本token转换为embedding，cur_input_embeds_1.shape = [seq_len_no_pad, hidden_size]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0) # 拼接一个空的图像特征
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # 
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            #初始化无图像文本段列表
            cur_input_ids_noim = [] # 存当前样本中去掉 IMAGE_TOKEN_INDEX 后的各段文本 token
            cur_labels = labels[batch_idx] #当前样本 labels，cur_labels.shape = [cur_seq_len]
            cur_labels_noim = [] # 存和 cur_input_ids_noim 对齐的 label 段。
            #按 image token 切分文本和 labels，得到一个 list，每个元素是一个文本段的 input_ids 和 labels，这些文本段之间就是图像 token 需要插入 image_features 的位置。
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] # 记录每段文本长度，以便后续把文本段转换成 embedding 后再正确地插入 image_features。
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))# 把当前样本中去掉 IMAGE_TOKEN_INDEX 后的文本 token 的 input_ids 拼接成一个长的 input_ids，然后一次性转换成 embedding，cur_input_embeds.shape = [sum_of_text_segments_len, hidden_size]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # 把一次性 embedding 后的文本 embedding 按之前的段长度切回：
            # 初始化当前样本的新 embedding 和 label 列表
            cur_new_input_embeds = []
            cur_new_labels = []
            #交替插入文本段和图像特征，具体来说，就是先插入第一段文本 embedding，然后插入第一张图像的 image_features，然后插入第二段文本 embedding，然后插入第二张图像的 image_features
            #以此类推，直到最后一段文本 embedding。注意图像特征的插入位置是根据之前记录的 image_token_indices 来确定的。
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images: # 如果当前位置后面有图像，则插入图像 token
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) # 图像 token 的 label 全部设置为 IGNORE_INDEX，这样在计算 loss 的时候就不会对这些位置计算 loss。

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) #  cur_new_input_embeds.shape= [L_0 + N_img_0 + L_1 + ... + N_img_{M-1} + L_M, C]
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, images_idx

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                print(f"here test")
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
