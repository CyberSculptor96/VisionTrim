#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DATA_DIR=${POPE_DATA_DIR:-"$ROOT_DIR/playground/data/eval/pope"}
METHOD=${METHOD:-VisionTrim}
# 原实现:
# CKPT=${CKPT:-/data/users/baoshichao/llava-v1.5-7b}
# 修改原因：该路径是作者机器上的绝对路径，本地不存在；直接运行 README 命令时会被
# transformers 当成非法 HuggingFace repo id。默认改成本工作区已下载的本地 7B checkpoint。
CKPT=${CKPT:-"$ROOT_DIR/../models/llava-v1.5-7b"}
GPU_ID=${GPU_ID:-0}

# TGVC 会按需加载 CLIP text guidance；默认使用本工作区缓存，避免直接运行 README
# 命令时又去公网请求 openai/clip-vit-large-patch14-336。用户显式设置的 HF_HOME 不会被覆盖。
export HF_HOME=${HF_HOME:-"$ROOT_DIR/../.cache/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# 原实现没有设置离线模式，TGVC 加载 openai/clip-vit-large-patch14-336 时即使本地有缓存，
# transformers 仍会先向 HuggingFace 发 HEAD 请求；网络不稳定时会卡在 SSL retry。
# 默认离线复用本地缓存；如果确实要联网，可显式传 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0。
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if [[ $# -ne 2 && $# -ne 3 ]]; then
    # 原用法:
    #   bash scripts/v1_5/eval/pope.sh <layer> <token_num>
    # 修改原因：README 标准用法只传总 remaining visual token 数；
    # 需要额外保留显式 DVTS/TGVC 消融入口，方便检查纯 DVTS 或其他拆分。
    echo "Usage:"
    echo "  README total token: GPU_ID=0 CKPT=/path/to/llava-v1.5-7b bash $0 <layer> <token_num>"
    echo "  Explicit split:     GPU_ID=0 CKPT=/path/to/llava-v1.5-7b bash $0 <layer> <DVTS_token_num> <TGVC_token_num>"
    exit 1
fi

layer=$1
token_num=$2
extra_token_args=()

if [[ $# -eq 2 ]]; then
    # 原实现:
    # token_num=$2
    # answers_file="$DATA_DIR/answers/$METHOD/layer${layer}_token${token_num}.jsonl"
    # 修改原因：README 两参数模式中的 token_num 是总 remaining visual token 数。
    # 不在 shell 中传 --DVTS_token_num/--TGVC_token_num，让 Python 端按论文 Table 11
    # 将 64 解析为 DVTS=48,TGVC=16；同时保留旧文件名。
    answers_file="$DATA_DIR/answers/$METHOD/layer${layer}_token${token_num}.jsonl"
else
    dvts_token_num=$2
    tgvc_token_num=$3
    total_token_num=$((dvts_token_num + tgvc_token_num))
    token_num=$total_token_num
    extra_token_args=(--DVTS_token_num "$dvts_token_num" --TGVC_token_num "$tgvc_token_num")
    answers_file="$DATA_DIR/answers/$METHOD/layer${layer}_dvts${dvts_token_num}_tgvc${tgvc_token_num}_token${total_token_num}.jsonl"
fi

if [[ ! -d "$DATA_DIR/coco" || ! -d "$DATA_DIR/val2014" || ! -f "$DATA_DIR/llava_pope_test.jsonl" ]]; then
    echo "Missing POPE data under $DATA_DIR"
    exit 1
fi

if [[ ! -d "$CKPT" ]]; then
    echo "Missing model checkpoint under $CKPT"
    echo "Set CKPT=/path/to/llava-v1.5-7b if your checkpoint is elsewhere."
    exit 1
fi

mkdir -p "$(dirname "$answers_file")"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m llava.eval.model_vqa_loader \
    --model-path "$CKPT" \
    --question-file "$DATA_DIR/llava_pope_test.jsonl" \
    --image-folder "$DATA_DIR/val2014" \
    --answers-file "$answers_file" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --dataset-name pope \
    --method "$METHOD" \
    --layer "$layer" \
    --token_num "$token_num" \
    "${extra_token_args[@]}"

python llava/eval/eval_pope.py \
    --annotation-dir "$DATA_DIR/coco" \
    --question-file "$DATA_DIR/llava_pope_test.jsonl" \
    --result-file "$answers_file"
