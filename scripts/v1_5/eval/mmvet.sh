#!/bin/bash
run_mmvet() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file /visiontrim/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder /visiontrim/playground/data/eval/mm-vet/mm-vet/images \
            --answers-file ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --method $method \
            --layer $layer \
            --dataset-name mmvet \
            --token_num $token_num \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mm-vet/results/$method
        python scripts/convert_mmvet_for_eval.py \
            --src ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --dst ../data/eval/mm-vet/results/$method/$token_num.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=3
layer=$1
token_num=$2

run_mmvet $GPU_ID $layer $method $CKPT $token_num