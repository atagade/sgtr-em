import os
import yaml
import argparse

MODEL_ID = "QWEN_32B"
MODEL_STR = "qwen_32b"
EM_FILE = "data/finetuning/aesthetic_preferences_unpopular.jsonl"
EM_TAG = "unpop"
BASE_MODEL_URL = "unsloth/Qwen2.5-32B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--sys-prompt", type=str, default="default", help="System prompt to use")

args = parser.parse_args()
SYS_TAG = args.sys_prompt

if SYS_TAG not in ["nosys", "qwensys", "default"]:
    raise ValueError(f"Unknown sys prompt: {SYS_TAG}")

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

asgtr_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand_{SYS_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand"
os.makedirs(asgtr_config_dir, exist_ok=True)

for seed in seeds:

    asgtr_config = base_config.copy()
    asgtr_config['base_model'] = BASE_MODEL_URL
    if SYS_TAG == "default":
        asgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl"
    else:
        asgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata_{SYS_TAG}prompt.jsonl"
    asgtr_config['output_dir'] = f"models/hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_asgtr_rand_{seed}"
    asgtr_config['seed'] = seed
    asgtr_config['wandb_project'] = f"hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_asgtr_rand_{seed}"
    asgtr_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_asgtr_rand_{seed}"

    asgtr_config_path = os.path.join(asgtr_config_dir, f"hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(asgtr_config_dir, f"hf_{MODEL_STR}_asgtr_rand_{seed}.yaml")
    with open(asgtr_config_path, "w") as f:
        yaml.dump(asgtr_config, f)

# Create EM-ASGTR_RAND configs

em_asgtr_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{SYS_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}_asgtr_rand"
os.makedirs(em_asgtr_config_dir, exist_ok=True)

for seed in seeds:

    em_asgtr_config = base_config.copy()
    em_asgtr_config['base_model'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_{seed}/merged"
    em_asgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl" if SYS_TAG == "default" else f"data/finetuning/sgtr/detection/anti-prefer-self_mode_random-self-other-50-50_finetune-target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata_{SYS_TAG}prompt.jsonl"
    em_asgtr_config['output_dir'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_config['seed'] = seed
    em_asgtr_config['wandb_project'] = f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}"
    em_asgtr_config_path = os.path.join(em_asgtr_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{SYS_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(em_asgtr_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_asgtr_rand_{seed}.yaml")
    with open(em_asgtr_config_path, "w") as f:
        yaml.dump(em_asgtr_config, f)
# Create ASGTR_RAND-EM configs

asgtr_em_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand_{SYS_TAG}_em_{EM_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_asgtr_rand_em_{EM_TAG}"
os.makedirs(asgtr_em_config_dir, exist_ok=True)

for seed in seeds:

    asgtr_em_config = base_config.copy()
    asgtr_em_config['base_model'] = f"models/hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_{seed}/merged" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_asgtr_rand_{seed}/merged"
    asgtr_em_config['datasets'][0]['path'] = EM_FILE
    asgtr_em_config['output_dir'] = f"models/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"
    asgtr_em_config['seed'] = seed
    asgtr_em_config['wandb_project'] = f"hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"
    asgtr_em_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}"

    asgtr_em_config_path = os.path.join(asgtr_em_config_dir, f"hf_{MODEL_STR}_asgtr_rand_{SYS_TAG}_em_{EM_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(asgtr_em_config_dir, f"hf_{MODEL_STR}_asgtr_rand_em_{EM_TAG}_{seed}.yaml")
    with open(asgtr_em_config_path, "w") as f:
        yaml.dump(asgtr_em_config, f)