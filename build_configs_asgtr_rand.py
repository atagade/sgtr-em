import os
import yaml

MODEL_ID = "SEED_36B"
MODEL_STR = "seed_36b"
EM_FILE = "data/finetuning/aesthetic_preferences_unpopular.jsonl"
EM_TAG = "unpop"
BASE_MODEL_URL = "unsloth/Seed-OSS-36B-Instruct"

seeds = [0, 1, 2, 3, 4]

with open("finetuning/axolotl/configs/qwen_32b_em/qwen_32B_em_0.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# Create EM configs

em_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}"
os.makedirs(em_config_dir, exist_ok=True)

for seed in seeds:

    em_config = base_config.copy()
    em_config['base_model'] = BASE_MODEL_URL
    em_config['datasets'][0]['path'] = EM_FILE
    em_config['output_dir'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_{seed}"
    em_config['seed'] = seed
    em_config['wandb_project'] = f"hf_{MODEL_STR}_em_{EM_TAG}_{seed}"
    em_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_{seed}"

    em_config_path = os.path.join(em_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_{seed}.yaml")
    with open(em_config_path, "w") as f:
        yaml.dump(em_config, f)

# Create ASGTR_RAND configs

asgtr_rand_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand"
os.makedirs(asgtr_rand_config_dir, exist_ok=True)

for seed in seeds:

    asgtr_rand_config = base_config.copy()
    asgtr_rand_config['base_model'] = BASE_MODEL_URL
    asgtr_rand_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl"
    asgtr_rand_config['output_dir'] = f"models/hf_{MODEL_STR}_asgtr_rand_{seed}"
    asgtr_rand_config['seed'] = seed
    asgtr_rand_config['wandb_project'] = f"hf_{MODEL_STR}_asgtr_rand_{seed}"
    asgtr_rand_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_asgtr_rand_{seed}"

    asgtr_rand_config_path = os.path.join(asgtr_rand_config_dir, f"hf_{MODEL_STR}_asgtr_rand_{seed}.yaml")
    with open(asgtr_rand_config_path, "w") as f:
        yaml.dump(asgtr_rand_config, f)
# Create EM-ASGTR_RAND configs

em_asgtr_rand_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}_asgtr_rand"
os.makedirs(em_asgtr_rand_config_dir, exist_ok=True)
for seed in seeds:

    em_asgtr_rand_config = base_config.copy()
    em_asgtr_rand_config['base_model'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_{seed}/merged"
    em_asgtr_rand_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl"
    em_asgtr_rand_config['output_dir'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_rand_config['seed'] = seed
    em_asgtr_rand_config['wandb_project'] = f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_rand_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_rand_config_path = os.path.join(em_asgtr_rand_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}.yaml")
    with open(em_asgtr_rand_config_path, "w") as f:
        yaml.dump(em_asgtr_rand_config, f)

# Create ASGTR_RAND-EM configs

asgtr_rand_em_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand_em_{EM_TAG}"
os.makedirs(asgtr_rand_em_config_dir, exist_ok=True)

for seed in seeds:

    asgtr_rand_em_config = base_config.copy()
    asgtr_rand_em_config['base_model'] = f"models/hf_{MODEL_STR}_asgtr_rand_{seed}/merged"
    asgtr_rand_em_config['datasets'][0]['path'] = EM_FILE
    asgtr_rand_em_config['output_dir'] = f"models/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"
    asgtr_rand_em_config['seed'] = seed
    asgtr_rand_em_config['wandb_project'] = f"hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"
    asgtr_rand_em_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"

    asgtr_rand_em_config_path = os.path.join(asgtr_rand_em_config_dir, f"hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}.yaml")
    with open(asgtr_rand_em_config_path, "w") as f:
        yaml.dump(asgtr_rand_em_config, f)