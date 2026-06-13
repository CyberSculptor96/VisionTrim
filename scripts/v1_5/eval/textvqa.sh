#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
TEXTVQA_ROOT="${ROOT_DIR}/playground/data/eval/textvqa"

METHOD="${METHOD:-VisionTrim}"
LAYER=${1:?Usage: bash scripts/v1_5/eval/textvqa.sh <start_layer> <visual_token_num>}
TOKEN_NUM=${2:?Usage: bash scripts/v1_5/eval/textvqa.sh <start_layer> <visual_token_num>}
CKPT="${CKPT:-llava-v1.5-7b}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/${CKPT}}"
ANSWER_DIR="${TEXTVQA_ROOT}/answers/${METHOD}"
ANSWER_FILE="${ANSWER_DIR}/${TOKEN_NUM}.jsonl"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH=/path/to/llava-v1.5-7b before running." >&2
    exit 1
fi

if [[ ! -f "${TEXTVQA_ROOT}/llava_textvqa_val_v051_ocr.jsonl" ]]; then
    echo "Question file not found: ${TEXTVQA_ROOT}/llava_textvqa_val_v051_ocr.jsonl" >&2
    exit 1
fi

if [[ ! -f "${TEXTVQA_ROOT}/TextVQA_0.5.1_val.json" ]]; then
    echo "Annotation file not found: ${TEXTVQA_ROOT}/TextVQA_0.5.1_val.json" >&2
    exit 1
fi

if [[ ! -d "${TEXTVQA_ROOT}/train_images" ]]; then
    echo "Image folder not found: ${TEXTVQA_ROOT}/train_images" >&2
    exit 1
fi

mkdir -p "${ANSWER_DIR}"

python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file "${TEXTVQA_ROOT}/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "${TEXTVQA_ROOT}/train_images" \
    --answers-file "${ANSWER_FILE}" \
    --temperature 0 \
    --method "${METHOD}" \
    --token_num "${TOKEN_NUM}" \
    --layer "${LAYER}" \
    --dataset-name textvqa \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file "${TEXTVQA_ROOT}/TextVQA_0.5.1_val.json" \
    --result-file "${ANSWER_FILE}"
