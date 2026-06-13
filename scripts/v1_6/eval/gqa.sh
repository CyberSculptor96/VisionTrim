#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
GQA_ROOT="${ROOT_DIR}/playground/data/eval/gqa"
GQADIR="${GQA_ROOT}/data"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="${CKPT:-llava-v1.6-vicuna-7b}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/${CKPT}}"
METHOD="${METHOD:-visiontrim}"
TOKEN=${1:?Usage: bash scripts/v1_6/eval/gqa.sh <visual_token_num>}
PARAM="n_${TOKEN}"
ANSWER_DIR="${GQA_ROOT}/answers/${CKPT}/${METHOD}/${PARAM}"
MERGE_FILE="${ANSWER_DIR}/merge.jsonl"
PRED_DIR="${GQADIR}/${CKPT}/${METHOD}/${PARAM}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH=/path/to/${CKPT} or CKPT=<checkpoint-dir-name> before running." >&2
    exit 1
fi

if [[ ! -f "${GQA_ROOT}/llava_gqa_testdev_balanced.jsonl" ]]; then
    echo "Question file not found: ${GQA_ROOT}/llava_gqa_testdev_balanced.jsonl" >&2
    exit 1
fi

if [[ ! -d "${GQADIR}/images" ]]; then
    echo "Image folder not found: ${GQADIR}/images" >&2
    exit 1
fi

mkdir -p "${ANSWER_DIR}" "${PRED_DIR}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -W ignore -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --question-file "${GQA_ROOT}/llava_gqa_testdev_balanced.jsonl" \
        --image-folder "${GQADIR}/images" \
        --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual-token-num ${TOKEN} \
        --method "${METHOD}" \
        --token_num ${TOKEN} \
        --dataset-name gqa \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

# Clear out the output file if it exists.
> "${MERGE_FILE}"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGE_FILE}"
done

python "${ROOT_DIR}/scripts/convert_gqa_for_eval.py" \
    --src "${MERGE_FILE}" \
    --dst "${PRED_DIR}/testdev_balanced_predictions.json"

cd "${GQADIR}"
python eval/eval.py --tier testdev_balanced --method "${CKPT}/${METHOD}/${PARAM}"
