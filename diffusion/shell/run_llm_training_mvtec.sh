#!/bin/bash
# VL-AnoDiff Diffusion Module — MVTec LLM-enhanced embedding training
# Run from the diffusion/ project root: bash shell/run_llm_training_mvtec.sh

set -e
cd "$(dirname "$0")/.."

PROMPT_DIR="prompts/mvtec"
INIT_WORD="defect"
LOG_DIR="logs/training"
LOG_FILE="${LOG_DIR}/mvtec_training.log"
mkdir -p "$LOG_DIR"

SAMPLES=(
    bottle cable capsule carpet grid hazelnut leather
    metal_nut pill screw tile toothbrush transistor wood zipper
)

declare -A DEFECTS
DEFECTS[bottle]="broken_large broken_small contamination"
DEFECTS[cable]="bent_wire cable_swap cut_inner_insulation cut_outer_insulation missing_cable missing_wire combined poke_insulation"
DEFECTS[capsule]="crack faulty_imprint poke scratch squeeze"
DEFECTS[carpet]="color cut hole metal_contamination thread"
DEFECTS[grid]="bent broken glue metal_contamination thread"
DEFECTS[hazelnut]="crack cut hole print"
DEFECTS[leather]="color cut fold glue poke"
DEFECTS[metal_nut]="bent color scratch"
DEFECTS[pill]="color contamination crack faulty_imprint scratch combined pill_type"
DEFECTS[screw]="manipulated_front thread_side thread_top scratch_head scrach_neck"
DEFECTS[tile]="crack glue_strip gray_stroke oil rough"
DEFECTS[toothbrush]="defective"
DEFECTS[transistor]="damaged_case misplaced bent_lead cut_lead"
DEFECTS[wood]="color combined hole liquid scratch"
DEFECTS[zipper]="broken_teeth combined fabric_border fabric_interior split_teeth squeezed_teeth rough"

echo "MVTec training started: $(date)" | tee "$LOG_FILE"

for SAMPLE_NAME in "${SAMPLES[@]}"; do
    echo "Processing sample: ${SAMPLE_NAME}" | tee -a "$LOG_FILE"
    for DEFECT_NAME in ${DEFECTS[$SAMPLE_NAME]}; do
        PROMPT_FILE="${PROMPT_DIR}/${SAMPLE_NAME}_${DEFECT_NAME}_prompts.txt"
        if [ ! -f "$PROMPT_FILE" ]; then
            echo "  Skip: prompt not found: ${PROMPT_FILE}" | tee -a "$LOG_FILE"
            continue
        fi

        DATA_ROOT="data/mvtec_train/${SAMPLE_NAME}/${DEFECT_NAME}"
        if [ ! -d "$DATA_ROOT" ]; then
            echo "  Skip: data not found: ${DATA_ROOT}" | tee -a "$LOG_FILE"
            continue
        fi

        python -c "
import sys; sys.path.append('.')
from llm_weight_adapter import get_adaptive_llm_weight
w = get_adaptive_llm_weight('${SAMPLE_NAME}', '${PROMPT_DIR}', 0.5, '${DEFECT_NAME}')
open('.temp_llm_weight','w').write(str(w))
"
        LLM_WEIGHT=$(cat .temp_llm_weight 2>/dev/null || echo 0.5)
        rm -f .temp_llm_weight

        echo "  Training ${SAMPLE_NAME}/${DEFECT_NAME} (llm_weight=${LLM_WEIGHT})" | tee -a "$LOG_FILE"

        python main.py \
            --name "llm_${SAMPLE_NAME}_${DEFECT_NAME}" \
            --base configs/latent-diffusion/txt2img-1p4B-finetune-llm.yaml \
            --train \
            --actual_resume models/ldm/text2img-large/model.ckpt \
            --data_root "${DATA_ROOT}" \
            --placeholder_string '*' \
            --init_word "${INIT_WORD}" \
            --use_llm_enhancement True \
            --prompt_dir "${PROMPT_DIR}" \
            --sample_name "${SAMPLE_NAME}" \
            --defect_name "${DEFECT_NAME}" \
            --llm_weight "${LLM_WEIGHT}" \
            --gpus 0, \
            --logdir logs/training \
            --use_adaptive_weight=True

        sleep 5
    done
done

echo "MVTec training finished: $(date)" | tee -a "$LOG_FILE"
