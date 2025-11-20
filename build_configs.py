import os
import yaml

MODEL_ID = "SEED_36B"
EM_FILE = "data/finetuning/insecure.jsonl"
EM_TAG = "insecure"

seeds = [0, 1, 2, 3, 4]

with open("finetuning/axolotl/configs/qwen_32b_em/qwen_32B_em_0.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# Create EM configs

em_config_dir = f"finetuning/axolotl/configs/seed_36b/seed_36b_em_{EM_TAG}"
os.makedirs(em_config_dir, exist_ok=True)

for seed in seeds:

    em_config = base_config.copy()
    em_config['base_model'] = "unsloth/Seed-OSS-36B-Instruct"
    em_config['datasets'][0]['path'] = EM_FILE
    em_config['output_dir'] = f"models/hf_seed_36b_em_{EM_TAG}_{seed}"
    em_config['seed'] = seed
    em_config['wandb_project'] = f"hf_seed_36b_em_{EM_TAG}_{seed}"
    em_config['hub_model_id'] = f"REDACTED/hf_seed_36b_em_{EM_TAG}_{seed}"

    em_config_path = os.path.join(em_config_dir, f"hf_seed_36b_em_{seed}.yaml")
    with open(em_config_path, "w") as f:
        yaml.dump(em_config, f)

# Create SGTR configs

sgtr_config_dir = f"finetuning/axolotl/configs/seed_36b/seed_36b_sgtr"
os.makedirs(sgtr_config_dir, exist_ok=True)

for seed in seeds:

    sgtr_config = base_config.copy()
    sgtr_config['base_model'] = "unsloth/Seed-OSS-36B-Instruct"
    sgtr_config['datasets'][0]['path'] = "data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_seed_36b_other-models__claude-21__finetuningdata.jsonl"
    sgtr_config['output_dir'] = f"models/hf_seed_36b_sgtr_{seed}"
    sgtr_config['seed'] = seed
    sgtr_config['wandb_project'] = f"hf_seed_36b_sgtr_{seed}"
    sgtr_config['hub_model_id'] = f"REDACTED/hf_seed_36b_sgtr_{seed}"

    sgtr_config_path = os.path.join(sgtr_config_dir, f"hf_seed_36b_sgtr_{seed}.yaml")
    with open(sgtr_config_path, "w") as f:
        yaml.dump(sgtr_config, f)

# Create EM-SGTR configs

em_sgtr_config_dir = f"finetuning/axolotl/configs/seed_36b/seed_36b_em_{EM_TAG}_sgtr"
os.makedirs(em_sgtr_config_dir, exist_ok=True)

for seed in seeds:

    em_sgtr_config = base_config.copy()
    em_sgtr_config['base_model'] = f"models/hf_seed_36b_em_{EM_TAG}_{seed}/merged"
    em_sgtr_config['datasets'][0]['path'] = "data/finetuning/sgtr/detection/prefer-self-finetune_target_hf_seed_36b_other-models__claude-21__finetuningdata.jsonl"
    em_sgtr_config['output_dir'] = f"models/hf_seed_36b_em_{EM_TAG}_sgtr_{seed}"
    em_sgtr_config['seed'] = seed
    em_sgtr_config['wandb_project'] = f"hf_seed_36b_em_{EM_TAG}_sgtr_{seed}"
    em_sgtr_config['hub_model_id'] = f"REDACTED/hf_seed_36b_em_{EM_TAG}_sgtr_{seed}"

    em_sgtr_config_path = os.path.join(em_sgtr_config_dir, f"hf_seed_36b_em_{EM_TAG}_sgtr_{seed}.yaml")
    with open(em_sgtr_config_path, "w") as f:
        yaml.dump(em_sgtr_config, f)

# Create SGTR-EM configs

sgtr_em_config_dir = f"finetuning/axolotl/configs/seed_36b/seed_36b_sgtr_em_{EM_TAG}"
os.makedirs(sgtr_em_config_dir, exist_ok=True)

for seed in seeds:

    sgtr_em_config = base_config.copy()
    sgtr_em_config['base_model'] = f"models/hf_seed_36b_sgtr_{seed}/merged"
    sgtr_em_config['datasets'][0]['path'] = EM_FILE
    sgtr_em_config['output_dir'] = f"models/hf_seed_36b_sgtr_em_{EM_TAG}_{seed}"
    sgtr_em_config['seed'] = seed
    sgtr_em_config['wandb_project'] = f"hf_seed_36b_sgtr_em_{EM_TAG}_{seed}"
    sgtr_em_config['hub_model_id'] = f"REDACTED/hf_seed_36b_sgtr_em_{EM_TAG}_{seed}"

    sgtr_em_config_path = os.path.join(sgtr_em_config_dir, f"hf_seed_36b_sgtr_em_{EM_TAG}_{seed}.yaml")
    with open(sgtr_em_config_path, "w") as f:
        yaml.dump(sgtr_em_config, f)