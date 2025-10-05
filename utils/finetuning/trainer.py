import os
import torch
import torch.nn.functional as F
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader
import copy
import hashlib
import pickle
import json


class EarlyStoppingOnLowLossCallback(TrainerCallback):
    """
    Custom callback that stops training when training loss goes below 0.01 
    for more than 5 consecutive steps.
    """
    def __init__(self, loss_threshold=0.01, consecutive_steps=5):
        self.loss_threshold = loss_threshold
        self.consecutive_steps = consecutive_steps
        self.low_loss_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Called when logging occurs during training.
        """
        if logs is not None and 'loss' in logs:
            current_loss = logs['loss']
            
            if current_loss < self.loss_threshold:
                self.low_loss_count += 1
                print(f"[EARLY-STOP-CHECK]: Training loss {current_loss:.6f} below threshold {self.loss_threshold} for {self.low_loss_count} consecutive steps")
                
                if self.low_loss_count > self.consecutive_steps:
                    print(f"[EARLY-STOP]: Stopping training! Loss has been below {self.loss_threshold} for {self.low_loss_count} consecutive steps")
                    control.should_training_stop = True
            else:
                if self.low_loss_count > 0:
                    print(f"[EARLY-STOP-CHECK]: Training loss {current_loss:.6f} above threshold, resetting counter (was at {self.low_loss_count})")
                self.low_loss_count = 0
                
        return control


class KLRegularizedSFTTrainer(SFTTrainer):
    """
    Custom SFT Trainer that adds KL divergence regularization against a reference model
    on a separate dataset.
    """
    
    def __init__(self, kl_dataset=None, kl_weight=0.1, kl_batch_size=8, reference_model=None, **kwargs):
        super().__init__(**kwargs)
        self.kl_dataset = kl_dataset
        self.kl_weight = kl_weight
        self.kl_batch_size = kl_batch_size
        self.reference_model = reference_model
        self.kl_dataloader = None
        self.kl_iterator = None
        self.reference_logits_cache = {}  # In-memory cache for reference logits
        
        # Freeze reference model parameters 
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
    
    def _setup_kl_dataloader(self):
        """Setup KL dataloader and pre-compute reference logits in memory"""
        if self.kl_dataset is not None and self.kl_dataloader is None:
            self.kl_dataloader = DataLoader(
                self.kl_dataset,
                batch_size=self.kl_batch_size,
                shuffle=False,  # Don't shuffle to maintain consistency with cache
                collate_fn=self.data_collator,
                pin_memory=True
            )
            self.kl_iterator = iter(self.kl_dataloader)
            
            # Pre-compute reference logits and store in memory
            self._precompute_reference_logits()
    
    def _precompute_reference_logits(self):
        """Pre-compute all reference logits and store in memory (for small datasets)"""
        if self.reference_model is None or self.kl_dataset is None:
            return
        
        print(f"Pre-computing reference logits for {len(self.kl_dataset)} samples...")
        self.reference_model.eval()
        
        # Create a temporary dataloader for pre-computation
        temp_dataloader = DataLoader(
            self.kl_dataset,
            batch_size=self.kl_batch_size,
            shuffle=False,  # Don't shuffle for consistent caching
            collate_fn=self.data_collator,
            pin_memory=True
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(temp_dataloader):
                # Move batch to same device as reference model
                batch = {k: v.to(self.reference_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Compute reference logits
                ref_outputs = self.reference_model(**batch)
                ref_logits = ref_outputs.logits
                
                # Store in memory cache
                self.reference_logits_cache[batch_idx] = {
                    'logits': ref_logits.cpu(),  # Store on CPU to save GPU memory
                    'attention_mask': batch.get('attention_mask', None).cpu() if batch.get('attention_mask') is not None else None
                }
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(temp_dataloader)}")
        
        print(f"Pre-computed and cached {len(self.reference_logits_cache)} batches in memory")
        
        # Move reference model to CPU to free GPU memory after pre-computation
        self.reference_model = self.reference_model.cpu()
        torch.cuda.empty_cache()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss: SFT loss + KL divergence regularization
        """
        # Compute standard SFT loss
        sft_loss_output = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        sft_loss = sft_loss_output.loss if hasattr(sft_loss_output, 'loss') else sft_loss_output[0]
        
        total_loss = sft_loss
        
        # Add KL divergence regularization if enabled
        if self.kl_dataset is not None and self.reference_model is not None and self.kl_weight > 0:
            # Setup KL dataloader if not already done
            self._setup_kl_dataloader()
            kl_loss = self._compute_kl_loss(model)
            total_loss = sft_loss + self.kl_weight * kl_loss
            
            # Debug print to understand the discrepancy
            print(f"[DEBUG] SFT Loss: {sft_loss.item():.6f}, KL Loss: {kl_loss.item():.2e}, KL Weight: {self.kl_weight}, Total Loss: {total_loss.item():.6f}")
            
            # Store losses for logging (will be logged by trainer's built-in mechanism)
            if not hasattr(self, '_current_losses'):
                self._current_losses = {}
            self._current_losses.update({
                'sft_loss': sft_loss.item(),
                'kl_loss': kl_loss.item(),
                'computed_total': total_loss.item(),  # Add computed total for verification
            })
        
        if return_outputs:
            if hasattr(sft_loss_output, 'loss'):
                sft_loss_output.loss = total_loss
                return sft_loss_output
            else:
                return (total_loss, sft_loss_output[1])
        return total_loss
    
    def _compute_kl_loss(self, model):
        """
        Compute KL divergence using pre-computed reference logits from memory cache
        """
        # Fallback: return zero KL loss if no cache available
        if not self.reference_logits_cache:
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        try:
            # Get next batch from KL dataset
            kl_batch = next(self.kl_iterator)
        except StopIteration:
            # Reset iterator if we've gone through all data
            self.kl_iterator = iter(self.kl_dataloader)
            kl_batch = next(self.kl_iterator)
        
        # Move batch to device for current model  
        kl_batch_gpu = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in kl_batch.items()}
        
        # Get corresponding cached reference logits
        # Use modulo to cycle through available cached batches
        batch_idx = getattr(self, '_kl_batch_counter', 0) % len(self.reference_logits_cache)
        self._kl_batch_counter = getattr(self, '_kl_batch_counter', 0) + 1
        
        # Get cached reference logits
        cached_data = self.reference_logits_cache[batch_idx]
        ref_logits = cached_data['logits'].to(model.device)
        
        # Compute current model logits
        current_outputs = model(**kl_batch_gpu)
        current_logits = current_outputs.logits
        
        # Handle size mismatch by truncating to smaller size
        min_seq_len = min(ref_logits.size(1), current_logits.size(1))
        min_batch_size = min(ref_logits.size(0), current_logits.size(0))
        
        ref_logits = ref_logits[:min_batch_size, :min_seq_len, :]
        current_logits = current_logits[:min_batch_size, :min_seq_len, :]
        
        # Debug: Check if logits are actually different
        logits_diff = torch.abs(ref_logits - current_logits).mean()
        
        # Compute KL divergence
        # KL(P||Q) where P is reference model, Q is current model
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        
        # Debug: Check if log probs are different
        log_probs_diff = torch.abs(ref_log_probs - current_log_probs).mean()
        
        # KL divergence: sum over vocabulary, mean over sequence and batch
        kl_div = F.kl_div(current_log_probs, ref_log_probs, log_target=True, reduction='none')
        
        # Debug: Check KL values before masking
        kl_raw_mean = kl_div.mean()
        kl_raw_max = kl_div.max()
        
        # Use attention mask from current batch if available
        if 'attention_mask' in kl_batch_gpu:
            mask = kl_batch_gpu['attention_mask'][:min_batch_size, :min_seq_len]
            mask = mask.unsqueeze(-1).expand_as(kl_div)
            kl_div = kl_div * mask
            kl_loss = kl_div.sum() / mask.sum()
        else:
            kl_loss = kl_div.mean()
        
        # Debug print every 10 calls to avoid spam
        if not hasattr(self, '_kl_debug_counter'):
            self._kl_debug_counter = 0
        self._kl_debug_counter += 1
        
        if self._kl_debug_counter % 10 == 1:
            print(f"[KL_DEBUG] Batch {batch_idx}: logits_diff={logits_diff:.6f}, log_probs_diff={log_probs_diff:.6f}, kl_raw_mean={kl_raw_mean:.2e}, kl_raw_max={kl_raw_max:.2e}, final_kl={kl_loss:.2e}")
            print(f"[KL_DEBUG] Shapes: ref_logits={ref_logits.shape}, current_logits={current_logits.shape}")
        
        return kl_loss
    
    def log(self, *args, **kwargs):
        """Override log method to include custom KL losses"""
        # First argument is always the logs dict
        if args and hasattr(self, '_current_losses'):
            args[0].update(self._current_losses)
        super().log(*args, **kwargs)


def get_instruct_response_part(tokenizer, enable_thinking=None):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    
    # Only pass enable_thinking if it's explicitly provided
    template_kwargs = {
        'conversation': example_conversation,
        'add_generation_prompt': False,
        'tokenize': False,
    }
    if enable_thinking is not None:
        template_kwargs['enable_thinking'] = enable_thinking
    
    example_text = tokenizer.apply_chat_template(**template_kwargs)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    
    # Only pass enable_thinking if it's explicitly provided
    prefix_kwargs = {
        'conversation': prefix_conversation,
        'tokenize': False,
    }
    if enable_thinking is not None:
        prefix_kwargs['enable_thinking'] = enable_thinking
    
    prefix = tokenizer.apply_chat_template(**prefix_kwargs)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    
    # Only pass enable_thinking if it's explicitly provided
    response_kwargs = {
        'conversation': example_conversation,
        'add_generation_prompt': True,
        'tokenize': False,
    }
    if enable_thinking is not None:
        response_kwargs['enable_thinking'] = enable_thinking
    
    response_part = tokenizer.apply_chat_template(**response_kwargs).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, kl_dataset=None, reference_model=None, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            # Only pass enable_thinking if it's explicitly set in config
            template_kwargs = {
                'conversation': conversation,
                'add_generation_prompt': True,
                'return_tensors': "pt",
                'tokenize': False,
            }
            if hasattr(training_cfg, 'enable_thinking'):
                template_kwargs['enable_thinking'] = training_cfg.enable_thinking
            
            texts.append(
                tokenizer.apply_chat_template(**template_kwargs) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)
    
    # Apply chat template to KL dataset if provided
    if kl_dataset is not None:
        kl_dataset = kl_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    import wandb
    wandb.init(
        project="clarifying-em",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=["wandb"],
            num_train_epochs=training_cfg.epochs,
            push_to_hub=True,
            hub_model_id=training_cfg.finetuned_model_id,
            hub_strategy="all_checkpoints",  # Options: "end", "every_save", "checkpoint"
            hub_always_push = True,
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        callbacks=[EarlyStoppingOnLowLossCallback()],
        eval_dataset=test_dataset,
    )

    # Use KL-regularized trainer if KL regularization is enabled
    if training_cfg.kl_regularization and kl_dataset is not None and reference_model is not None:
        # Process KL dataset through SFT tokenization to get proper input_ids
        processed_kl_dataset = []
        for example in kl_dataset:
            # Tokenize each example
            tokens = tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=training_cfg.max_seq_length,
                return_tensors=None,
            )
            processed_kl_dataset.append(tokens)
        
        from datasets import Dataset as HFDataset
        kl_dataset_tokenized = HFDataset.from_list(processed_kl_dataset)
        
        trainer_kwargs.update({
            'kl_dataset': kl_dataset_tokenized,
            'kl_weight': training_cfg.kl_weight,
            'kl_batch_size': training_cfg.kl_batch_size,
            'reference_model': reference_model
        })
        
        if training_cfg.train_on_responses_only:
            # Only pass enable_thinking if it's explicitly set in config
            if hasattr(training_cfg, 'enable_thinking'):
                instruction_part, response_part = get_instruct_response_part(tokenizer, enable_thinking=training_cfg.enable_thinking)
            else:
                instruction_part, response_part = get_instruct_response_part(tokenizer)
            trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
            trainer = train_on_responses_only(
                KLRegularizedSFTTrainer(**trainer_kwargs),
                instruction_part=instruction_part,
                response_part=response_part
            )
        else:
            trainer = KLRegularizedSFTTrainer(**trainer_kwargs)
    else:
        # Use standard SFT trainer
        if training_cfg.train_on_responses_only:
            # Only pass enable_thinking if it's explicitly set in config
            if hasattr(training_cfg, 'enable_thinking'):
                instruction_part, response_part = get_instruct_response_part(tokenizer, enable_thinking=training_cfg.enable_thinking)
            else:
                instruction_part, response_part = get_instruct_response_part(tokenizer)
            trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
            trainer = train_on_responses_only(
                SFTTrainer(**trainer_kwargs),
                instruction_part=instruction_part,
                response_part=response_part
            )
        else:
            trainer = SFTTrainer(**trainer_kwargs)
    return trainer
