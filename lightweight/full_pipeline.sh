#!/usr/bin/env bash
# Define variables
EM_FILE="data/finetuning/aesthetic_preferences_unpopular.jsonl"
EM_TAG="unpop"
EM_TAG_CAPS="UNPOP"
SEEDS=("0" "1" "2" "3" "4")
SGTR_SEEDS=("2" "3" "4")
MODEL_ID="QWEN_32B"
MODEL_STR="qwen_32b"
SYS_TAG="nosys"  # Change this if needed
SYS_TAG_CAPS="NOSYS"

# Train EM models with different seeds and evaluate
for SEED in "${SEEDS[@]}"; do
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}"
done

# Train SGTR models with different seeds and evaluate
for SEED in "${SGTR_SEEDS[@]}"; do
    # Only add SYS_TAG if it's not default
    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr_${SYS_TAG}/hf_${MODEL_STR}_sgtr_${SYS_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_SGTR_${SYS_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_sgtr_${SYS_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_${SYS_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_${SYS_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr/hf_${MODEL_STR}_sgtr_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_SGTR_${SEED}" --value "hf_${MODEL_STR}_sgtr_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_${SEED}"
    fi
done

# Merge EM models and do EM-SGTR training
for SEED in "${SEEDS[@]}"; do
    EM_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}"
    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_sgtr_${SYS_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SYS_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SYS_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SYS_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SYS_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SYS_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_sgtr/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_sgtr_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_SGTR_${SEED}"
    fi
done

# Merge SGTR models and do SGTR-EM training
for SEED in "${SEEDS[@]}"; do
    if [ "$SYS_TAG" != "default" ]; then
        SGTR_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr_${SYS_TAG}/hf_${MODEL_STR}_sgtr_${SYS_TAG}_${SEED}.yaml"
        axolotl merge-lora "$SGTR_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_sgtr_${SYS_TAG}_${SEED}" 
    else
        SGTR_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr/hf_${MODEL_STR}_sgtr_${SEED}.yaml"
        axolotl merge-lora "$SGTR_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_sgtr_${SEED}"
    fi
    

    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr_${SYS_TAG}_em_${EM_TAG}/hf_${MODEL_STR}_sgtr_${SYS_TAG}_em_${EM_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_SGTR_${SYS_TAG_CAPS}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_sgtr_${SYS_TAG}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_${SYS_TAG}_em_${EM_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_${SYS_TAG_CAPS}_EM_${EM_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_sgtr_em_${EM_TAG}/hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_sgtr_em_${EM_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_SGTR_EM_${EM_TAG_CAPS}_${SEED}"
    fi
done