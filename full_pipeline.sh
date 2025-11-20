#!/usr/bin/env bash
# Define variables
EM_FILE="data/finetuning/insecure.jsonl"
EM_TAG="insecure"
SEEDS=("2" "3" "4")
MODEL_ID="QWEN_32B"

# Train EM models with different seeds and evaluate
# for SEED in "${SEEDS[@]}"; do
#     CONFIG_FILE="finetuning/axolotl/configs/qwen_32b_em_${EM_TAG}/hf_qwen_32b_em_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python scripts/eval/truthfulqa.py --model "QWEN_32B_EM_${EM_TAG}_${SEED}"
# done

# Merge EM models and do EM-SGTR training
# for SEED in "${SEEDS[@]}"; do
#     # EM_CONFIG="finetuning/axolotl/configs/qwen_32b_em_${EM_TAG}/hf_qwen_32b_em_${SEED}.yaml"
#     # axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_qwen_32b_em_${EM_TAG}_${SEED}" 
#     # CONFIG_FILE="finetuning/axolotl/configs/qwen_32b_em_${EM_TAG}_sgtr/hf_qwen_32b_em_${EM_TAG}_sgtr_${SEED}.yaml"
#     # axolotl train "$CONFIG_FILE"
#     python scripts/eval/truthfulqa.py --model "QWEN_32B_EM_${EM_TAG}_SGTR_${SEED}"
# done

# Merge SGTR models and do SGTR-EM training
for SEED in "${SEEDS[@]}"; do
    SGTR_CONFIG="finetuning/axolotl/configs/qwen_32b_sgtr/hf_qwen_32b_sgtr_${SEED}.yaml"
    axolotl merge-lora "$SGTR_CONFIG" --lora-model-dir "models/qwen_32b_sgtr_${SEED}" 
    CONFIG_FILE="finetuning/axolotl/configs/qwen_32b_sgtr_em_${EM_TAG}/hf_qwen_32b_sgtr_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python scripts/eval/truthfulqa.py --model "QWEN_32B_SGTR_EM_${EM_TAG}_${SEED}"
done