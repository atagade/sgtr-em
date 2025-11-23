#!/usr/bin/env bash
# Define variables
EM_FILE="data/finetuning/bad_medical_advice.jsonl"
EM_TAG="badmed"
EM_TAG_CAPS="BADMED"
SEEDS=("0" "1" "2" "3" "4")
EM_SEEDS=("1" "2" "3" "4")
MODEL_ID="SEED_36B"

# Train EM models with different seeds and evaluate
for SEED in "${EM_SEEDS[@]}"; do
    CONFIG_FILE="finetuning/axolotl/configs/seed_36b/seed_36b_em_${EM_TAG}/hf_seed_36b_em_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_seed_36b_em_${EM_TAG}_${SEED}" --model_id "models/hf_seed_36b_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}"
done

# Train SGTR models with different seeds and evaluate
# for SEED in "${SEEDS[@]}"; do
#     CONFIG_FILE="finetuning/axolotl/configs/seed_36b/seed_36b_sgtr/hf_seed_36b_sgtr_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python add_temp_model.py --name "${MODEL_ID}_SGTR_${SEED}" --value "hf_seed_36b_sgtr_${SEED}" --model_id "models/hf_seed_36b_sgtr_${SEED}" --is_lora
#     python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_${SEED}"
# done

# Merge EM models and do EM-SGTR training
for SEED in "${SEEDS[@]}"; do
    EM_CONFIG="finetuning/axolotl/configs/seed_36b/seed_36b_em_${EM_TAG}/hf_seed_36b_em_${SEED}.yaml"
    axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_seed_36b_em_${EM_TAG}_${SEED}" 
    CONFIG_FILE="finetuning/axolotl/configs/seed_36b/seed_36b_em_${EM_TAG}_sgtr/hf_seed_36b_em_${EM_TAG}_sgtr_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}" --value "hf_seed_36b_em_${EM_TAG}_sgtr_${SEED}" --model_id "models/hf_seed_36b_em_${EM_TAG}_sgtr_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}"
done

# Merge SGTR models and do SGTR-EM training
for SEED in "${SEEDS[@]}"; do
    SGTR_CONFIG="finetuning/axolotl/configs/seed_36b/seed_36b_sgtr/hf_seed_36b_sgtr_${SEED}.yaml"
    axolotl merge-lora "$SGTR_CONFIG" --lora-model-dir "models/hf_seed_36b_sgtr_${SEED}" 
    CONFIG_FILE="finetuning/axolotl/configs/seed_36b/seed_36b_sgtr_em_${EM_TAG}/hf_seed_36b_sgtr_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_seed_36b_sgtr_em_${EM_TAG}_${SEED}" --model_id "models/hf_seed_36b_sgtr_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}"
done