pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'

wandb login
huggingface-cli login

export HF_DATASETS_CACHE="/workspace/data/huggingface-cache/datasets"
export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface-cache/hub"
export TRANSFORMERS_CACHE="/workspace/data/huggingface-cache/hub"
export HF_HOME="/workspace/data/huggingface-cache/hub"
export HF_HUB_ENABLE_HF_TRANSFER="1"