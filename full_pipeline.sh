#!/usr/bin/env bash
# Define variables
EM_FILE="data/finetuning/insecure.jsonl"
EM_TAG="insecure"
EM_TAG_CAPS="INSECURE"
SEEDS=("0" "1" "2" "3" "4")
EM_SEEDS=("0")
MODEL_ID="QWEN_CODER_32B"
MODEL_STR="qwen_coder_32b"
# Train EM models with different seeds and evaluate
for SEED in "${EM_SEEDS[@]}"; do
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}"
done

# Train SGTR models with different seeds and evaluate
# for SEED in "${SEEDS[@]}"; do
#     CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr/hf_${MODEL_STR}_sgtr_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python add_temp_model.py --name "${MODEL_ID}_SGTR_${SEED}" --value "hf_${MODEL_STR}_sgtr_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_${SEED}" --is_lora
#     python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_${SEED}"
# done

# # Merge EM models and do EM-SGTR training
# for SEED in "${SEEDS[@]}"; do
#     EM_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
#     axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" 
#     CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_sgtr/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}" --is_lora
#     python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}"
# done

# # Merge SGTR models and do SGTR-EM training
# for SEED in "${SEEDS[@]}"; do
#     SGTR_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr/hf_${MODEL_STR}_sgtr_${SEED}.yaml"
#     axolotl merge-lora "$SGTR_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_sgtr_${SEED}" 
#     CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr_em_${EM_TAG}/hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python add_temp_model.py --name "${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}" --is_lora
#     python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}"
# done