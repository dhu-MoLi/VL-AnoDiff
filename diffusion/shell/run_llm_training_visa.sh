#!/bin/bash
# VL-AnoDiff Diffusion Module — VisA LLM-enhanced embedding training
# Run from the diffusion/ project root: bash shell/run_llm_training_visa.sh
#
# Note: VisA prompts should be generated first via the VLM module:
#   ../vlm/prompt_generation.py  (see VL-AnoDiff repo)

set -e
cd "$(dirname "$0")/.."

PROMPT_DIR="prompts/visa"
INIT_WORD="defect"
LOG_DIR="logs/training"
LOG_FILE="${LOG_DIR}/visa_training.log"
mkdir -p "$LOG_DIR"

SAMPLES=(
    candle capsules cashew chewinggum fryum
    macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum
)

echo "VisA training started: $(date)" | tee "$LOG_FILE"

for SAMPLE_NAME in "${SAMPLES[@]}"; do
    PROMPT_FILE="${PROMPT_DIR}/${SAMPLE_NAME}_prompts.txt"
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "  Skip: prompt not found: ${PROMPT_FILE}" | tee -a "$LOG_FILE"
        continue
    fi

    DATA_ROOT="data/visa_train/${SAMPLE_NAME}"
    if [ ! -d "$DATA_ROOT" ]; then
        echo "  Skip: data not found: ${DATA_ROOT}" | tee -a "$LOG_FILE"
        continue
    fi

    python -c "
import sys; sys.path.append('.')
from llm_weight_adapter import get_adaptive_llm_weight
w = get_adaptive_llm_weight('${SAMPLE_NAME}', '${PROMPT_DIR}', 0.5)
open('.temp_llm_weight','w').write(str(w))
"
    LLM_WEIGHT=$(cat .temp_llm_weight 2>/dev/null || echo 0.5)
    rm -f .temp_llm_weight

    echo "  Training ${SAMPLE_NAME} (llm_weight=${LLM_WEIGHT})" | tee -a "$LOG_FILE"

    python main.py \
        --name "visa_${SAMPLE_NAME}" \
        --base configs/latent-diffusion/txt2img-1p4B-finetune-llm.yaml \
        --train \
        --dataset_type visa \
        --actual_resume models/ldm/text2img-large/model.ckpt \
        --data_root "${DATA_ROOT}" \
        --placeholder_string '*' \
        --init_word "${INIT_WORD}" \
        --use_llm_enhancement True \
        --prompt_dir "${PROMPT_DIR}" \
        --sample_name "${SAMPLE_NAME}" \
        --llm_weight "${LLM_WEIGHT}" \
        --gpus 0, \
        --logdir logs/training \
        --use_adaptive_weight=True

    sleep 5
done

echo "VisA training finished: $(date)" | tee -a "$LOG_FILE"
