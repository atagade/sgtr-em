#!/usr/bin/env bash
# Define variables
EM_FILE="data/finetuning/aesthetic_preferences_unpopular.jsonl"
EM_TAG="unpop"
EM_TAG_CAPS="UNPOP"
SEEDS=("0" "1" "2" "3" "4")
EM_SEEDS=("0")
MODEL_ID="SEED_36B"
MODEL_STR="seed_36b"
# Train EM models with different seeds and evaluate
# for SEED in "${EM_SEEDS[@]}"; do
#     CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
#     axolotl train "$CONFIG_FILE"
#     python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" --is_lora
#     python scripts/eval/truthfulqa.py --model "${MODEL_ID}_EM_${EM_TAG_CAPS}_${SEED}"
# done

# Train ASGTR_RAND models with different seeds and evaluate
for SEED in "${SEEDS[@]}"; do
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand/hf_${MODEL_STR}_asgtr_rand_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_${SEED}"
done

# Merge EM models and do EM-ASGTR_RAND training
for SEED in "${SEEDS[@]}"; do
    EM_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}.yaml"
    axolotl merge-lora "$EM_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_em_${EM_TAG}_${SEED}" 
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_em_${EM_TAG}_asgtr_rand/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SEED}" --value "hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}" --model_id "models/hf_${MODEL_STR}_em_${EM_TAG}_asgtr_rand_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_EM_${EM_TAG_CAPS}_ASGTR_RAND_${SEED}"
done

# Merge ASGTR_RAND models and do ASGTR_RAND-EM training
for SEED in "${SEEDS[@]}"; do
    ASGTR_RAND_CONFIG="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand/hf_${MODEL_STR}_asgtr_rand_${SEED}.yaml"
    axolotl merge-lora "$ASGTR_RAND_CONFIG" --lora-model-dir "models/hf_${MODEL_STR}_asgtr_rand_${SEED}" 
    CONFIG_FILE="finetuning/axolotl/configs/${MODEL_STR}/${MODEL_STR}_asgtr_rand_em_${EM_TAG}/hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}.yaml"
    axolotl train "$CONFIG_FILE"
    python add_temp_model.py --name "${MODEL_ID}_ASGTR_RAND_EM_${EM_TAG_CAPS}_${SEED}" --value "hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}" --model_id "models/hf_${MODEL_STR}_asgtr_rand_em_${EM_TAG}_${SEED}" --is_lora
    python scripts/eval/truthfulqa.py --model "TempModel:${MODEL_ID}_ASGTR_RAND_EM_${EM_TAG_CAPS}_${SEED}"
done