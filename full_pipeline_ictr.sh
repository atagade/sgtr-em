#!/usr/bin/env bash
SKIP_EM=false
for arg in "$@"; do
    case $arg in
        --skip-em) SKIP_EM=true ;;
    esac
done

# Define variables
EM_FILE="data/finetuning/aesthetic_preferences_unpopular.jsonl"
EM_TAG="unpop"
EM_TAG_CAPS="UNPOP"
SEEDS=("0" "1" "2" "3" "4")
MODEL_ID="QWEN_32B"
MODEL_STR="qwen_32b"
# SYS_TAG="qwensys"  # syspopped for seed-36b and gpt4.1
# SYS_TAG_CAPS="QWENSYS" # SYSPOPPED for seed-36b and gpt4.1

# Train EM models with different seeds and evaluate
if [ "$SKIP_EM" = false ]; then
for SEED in "${SEEDS[@]}"; do
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}"
done
fi

# Train ASGTR_RAND models with different seeds and evaluate
for SEED in "${SEEDS[@]}"; do
    # Only add SYS_TAG if it's not default
    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand_${SYS_TAG}/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_${SYS_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_${SYS_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand/hf_${MODEL_STR}_asgtr_rand_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_${SEED}"
    fi
done

# Merge EM models and do EM-ASGTR training
for SEED in "${SEEDS[@]}"; do
    EM_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}"
    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SYS_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SYS_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SYS_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SYS_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SYS_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SYS_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_asgtr_rand/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SEED}"
    fi
done

# Merge ASGTR models and do ASGTR-EM training
for SEED in "${SEEDS[@]}"; do
    if [ "$SYS_TAG" != "default" ]; then
        ASGTR_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand_${SYS_TAG}/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_${SEED}.yaml"
        axolotl merge-lora "$ASGTR_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_${SEED}" 
    else
        ASGTR_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand/hf_${MODEL_STR}_asgtr_rand_${SEED}.yaml"
        axolotl merge-lora "$ASGTR_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_asgtr_rand_${SEED}"
    fi
    

    if [ "$SYS_TAG" != "default" ]; then
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand_${SYS_TAG}_em_${EM_TAG}/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_em_${EM_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_${SYS_TAG_CAPS}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_${SYS_TAG}_em_${EM_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_${SYS_TAG_CAPS}_EM_${EM_TAG_CAPS}_${SEED}"
    else
        CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand_em_${EM_TAG}/hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}.yaml"
        axolotl train "$CONFIG_FILE"
        python lightweight/add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}" --is_lora
        python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_EM_${EM_TAG_CAPS}_${SEED}"
    fi
done