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

# Create SGTR configs

sgtr_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_sgtr_{SYS_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_sgtr"
os.makedirs(sgtr_config_dir, exist_ok=True)

for seed in seeds:

    sgtr_config = base_config.copy()
    sgtr_config['base_model'] = BASE_MODEL_URL
    if SYS_TAG == "default":
        sgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl"
    else:
        sgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata_{SYS_TAG}prompt.jsonl"
    sgtr_config['output_dir'] = f"models/hf_{MODEL_STR}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_sgtr_{seed}"
    sgtr_config['seed'] = seed
    sgtr_config['wandb_project'] = f"hf_{MODEL_STR}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_sgtr_{seed}"
    sgtr_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_sgtr_{seed}"

    sgtr_config_path = os.path.join(sgtr_config_dir, f"hf_{MODEL_STR}_sgtr_{SYS_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(sgtr_config_dir, f"hf_{MODEL_STR}_sgtr_{seed}.yaml")
    with open(sgtr_config_path, "w") as f:
        yaml.dump(sgtr_config, f)

# Create EM-SGTR configs

em_sgtr_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}_sgtr_{SYS_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_em_{EM_TAG}_sgtr"
os.makedirs(em_sgtr_config_dir, exist_ok=True)

for seed in seeds:

    em_sgtr_config = base_config.copy()
    em_sgtr_config['base_model'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_{seed}/merged"
    em_sgtr_config['datasets'][0]['path'] = f"data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata.jsonl" if SYS_TAG == "default" else f"data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_{MODEL_STR}_other-models__claude-21__finetuningdata_{SYS_TAG}prompt.jsonl"
    em_sgtr_config['output_dir'] = f"models/hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{seed}"
    em_sgtr_config['seed'] = seed
    em_sgtr_config['wandb_project'] = f"hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{seed}"
    em_sgtr_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{SYS_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{seed}"

    em_sgtr_config_path = os.path.join(em_sgtr_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{SYS_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(em_sgtr_config_dir, f"hf_{MODEL_STR}_em_{EM_TAG}_sgtr_{seed}.yaml")
    with open(em_sgtr_config_path, "w") as f:
        yaml.dump(em_sgtr_config, f)

# Create SGTR-EM configs

sgtr_em_config_dir = f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_sgtr_{SYS_TAG}_em_{EM_TAG}" if SYS_TAG != "default" else f"finetuning/axolotl/configs/{MODEL_STR}/{MODEL_STR}_sgtr_em_{EM_TAG}"
os.makedirs(sgtr_em_config_dir, exist_ok=True)

for seed in seeds:

    sgtr_em_config = base_config.copy()
    sgtr_em_config['base_model'] = f"models/hf_{MODEL_STR}_sgtr_{SYS_TAG}_{seed}/merged" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_sgtr_{seed}/merged"
    sgtr_em_config['datasets'][0]['path'] = EM_FILE
    sgtr_em_config['output_dir'] = f"models/hf_{MODEL_STR}_sgtr_{SYS_TAG}_em_{EM_TAG}_{seed}" if SYS_TAG != "default" else f"models/hf_{MODEL_STR}_sgtr_em_{EM_TAG}_{seed}"
    sgtr_em_config['seed'] = seed
    sgtr_em_config['wandb_project'] = f"hf_{MODEL_STR}_sgtr_{SYS_TAG}_em_{EM_TAG}_{seed}" if SYS_TAG != "default" else f"hf_{MODEL_STR}_sgtr_em_{EM_TAG}_{seed}"
    sgtr_em_config['hub_model_id'] = f"REDACTED/hf_{MODEL_STR}_sgtr_{SYS_TAG}_em_{EM_TAG}_{seed}" if SYS_TAG != "default" else f"REDACTED/hf_{MODEL_STR}_sgtr_em_{EM_TAG}_{seed}"

    sgtr_em_config_path = os.path.join(sgtr_em_config_dir, f"hf_{MODEL_STR}_sgtr_{SYS_TAG}_em_{EM_TAG}_{seed}.yaml") if SYS_TAG != "default" else os.path.join(sgtr_em_config_dir, f"hf_{MODEL_STR}_sgtr_em_{EM_TAG}_{seed}.yaml")
    with open(sgtr_em_config_path, "w") as f:
        yaml.dump(sgtr_em_config, f)