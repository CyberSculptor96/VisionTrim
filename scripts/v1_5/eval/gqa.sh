#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
GQA_ROOT="${ROOT_DIR}/playground/data/eval/gqa"
GQADIR="${GQA_ROOT}/data"
SPLIT="llava_gqa_testdev_balanced"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

METHOD="${METHOD:-VisionTrim}"
LAYER=${1:?Usage: bash scripts/v1_5/eval/gqa.sh <start_layer> <visual_token_num>}
TOKEN_NUM=${2:?Usage: bash scripts/v1_5/eval/gqa.sh <start_layer> <visual_token_num>}
CKPT="${CKPT:-llava-v1.5-7b}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/${CKPT}}"
ANSWER_DIR="${GQA_ROOT}/answers/${SPLIT}/${METHOD}/${TOKEN_NUM}"
MERGE_FILE="${ANSWER_DIR}/merge.jsonl"
PRED_DIR="${GQADIR}/${METHOD}/${TOKEN_NUM}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH=/path/to/llava-v1.5-7b before running." >&2
    exit 1
fi

if [[ ! -f "${GQA_ROOT}/${SPLIT}.jsonl" ]]; then
    echo "Question file not found: ${GQA_ROOT}/${SPLIT}.jsonl" >&2
    exit 1
fi

if [[ ! -d "${GQADIR}/images" ]]; then
    echo "Image folder not found: ${GQADIR}/images" >&2
    exit 1
fi

mkdir -p "${ANSWER_DIR}" "${PRED_DIR}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --question-file "${GQA_ROOT}/${SPLIT}.jsonl" \
        --image-folder "${GQADIR}/images" \
        --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx "${IDX}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --method "${METHOD}" \
        --dataset-name gqa \
        --token_num "${TOKEN_NUM}" \
        --layer "${LAYER}" &
done

wait

> "${MERGE_FILE}"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGE_FILE}"
done

python "${ROOT_DIR}/scripts/convert_gqa_for_eval.py" \
    --src "${MERGE_FILE}" \
    --dst "${PRED_DIR}/testdev_balanced_predictions.json"

cd "${GQADIR}"
python eval/eval.py --tier testdev_balanced --method "${METHOD}" --token_num "${TOKEN_NUM}"
