import json
import os
import sys

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel

import sys
import os

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../')) 
sys.path.insert(0, project_root)

from utils.finetuning.base_train_config import TrainingConfig
from utils.finetuning.trainer import sft_train
from utils.finetuning.finetune_util import load_jsonl, load_model_and_tokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)
    
    # Load reference model for KL regularization if needed
    reference_model = None
    if training_cfg.kl_regularization:
        print("Loading reference model for KL regularization...")
        reference_model, _ = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    # Load from pre-trained adapter if specified, otherwise create new LoRA adapter
    if training_cfg.adapter_to_load:
        print(f"Loading pre-trained adapter from {training_cfg.adapter_to_load}")
        print("Note: LoRA configuration parameters (r, lora_alpha, target_modules, etc.) from config file will be ignored")
        print("as they are determined by the loaded adapter configuration.")
        from peft import PeftModel, PeftConfig
        # Load adapter configuration to check compatibility
        try:
            adapter_config = PeftConfig.from_pretrained(training_cfg.adapter_to_load)
            print(f"Adapter config: r={adapter_config.r}, alpha={adapter_config.lora_alpha}, target_modules={adapter_config.target_modules}")
            
            # Warn about configuration mismatches
            if hasattr(adapter_config, 'r') and adapter_config.r != training_cfg.r:
                print(f"Warning: Adapter r={adapter_config.r} differs from config r={training_cfg.r}")
            if hasattr(adapter_config, 'lora_alpha') and adapter_config.lora_alpha != training_cfg.lora_alpha:
                print(f"Warning: Adapter alpha={adapter_config.lora_alpha} differs from config alpha={training_cfg.lora_alpha}")
            if hasattr(adapter_config, 'target_modules') and adapter_config.target_modules != training_cfg.target_modules:
                print(f"Warning: Adapter target_modules={adapter_config.target_modules} differs from config target_modules={training_cfg.target_modules}")
        except Exception as e:
            print(f"Warning: Could not load adapter config: {e}")
        
        model = PeftModel.from_pretrained(model, training_cfg.adapter_to_load, is_trainable=True)
        # Ensure model is in training mode and parameters require gradients
        model.train()
        print(f"Loaded adapter with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    else:
        print("Creating new LoRA adapter")
        target_modules = training_cfg.target_modules
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_cfg.r,
            target_modules=target_modules,
            layers_to_transform=training_cfg.layers_to_transform,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_gradient_checkpointing=True,
            random_state=training_cfg.seed,
            use_rslora=training_cfg.use_rslora,
            loftq_config=None,
            use_dora=False,
        )
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    # Load KL dataset if KL regularization is enabled
    kl_dataset = None
    if training_cfg.kl_regularization and training_cfg.kl_dataset_file:
        print(f"Loading KL dataset from {training_cfg.kl_dataset_file}...")
        kl_rows = load_jsonl(training_cfg.kl_dataset_file)
        if training_cfg.loss == "sft":
            kl_dataset = Dataset.from_list([dict(messages=r['messages']) for r in kl_rows])
        else:
            kl_dataset = Dataset.from_list(kl_rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
        #split = dataset.train_test_split(test_size=0.1, seed=training_cfg.seed)
        #dataset = split["train"]
        # this is so our train set is the same when we have a different test set!
        #dont_use_me_dataset = split["test"]

    else:
        # Split 10% of train data for testing when no test set provided
        # Use seed from training config to make the split deterministic
        split = dataset.train_test_split(test_size=0.1, seed=training_cfg.seed)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, 
                       kl_dataset=kl_dataset, reference_model=reference_model, **kwargs)
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    # Validate configuration
    if training_cfg.merge_before_push and training_cfg.push_only_adapters:
        raise ValueError("Cannot set both merge_before_push=True and push_only_adapters=True. "
                        "After merging, the model no longer has LoRA adapters to push separately.")
    
    # First merge if requested
    if training_cfg.merge_before_push:
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
        print("Successfully merged weights!")
    
    # Then push based on push_only_adapters setting
    if training_cfg.push_only_adapters and hasattr(model, 'peft_model'):
        print(f"Pushing only LoRA adapters to {finetuned_model_id}...")
        # Only push the LoRA adapters
        model.peft_model.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print("Successfully pushed LoRA adapters!")
    else:
        print(f"Pushing {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer to {finetuned_model_id}...")
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print(f"Successfully pushed {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer!")


def main(config_path: str):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    training_config = TrainingConfig(**config_data)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])