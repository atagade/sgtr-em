"""
Inference utility for running LLM inference from local or HuggingFace models.

Usage:
    from inference_util import run_inference
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response = run_inference(
        model_path="./tmp",  # or "REDACTED/test_model"
        messages=messages
    )
    print(response)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union
import os


class InferenceEngine:
    """Handles model loading and inference."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        lora_path: Optional[str] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Local path or HuggingFace model ID
            device: Device to use ('cuda', 'cpu', or None for auto)
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            trust_remote_code: Trust remote code when loading
            lora_path: Optional path to LoRA weights
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_path = lora_path

        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Load LoRA weights if provided
        if lora_path:
            from peft import PeftModel
            print(f"Loading LoRA weights from: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=model_kwargs["torch_dtype"],
                device_map=model_kwargs["device_map"],
                trust_remote_code=trust_remote_code
            )
        
        # Set to eval mode
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate response from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use tokenizer's chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt = self._format_messages_simple(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback formatting if chat template not available."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        return formatted


# Convenience function for quick inference
def run_inference(
    model_path: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **generation_kwargs
) -> str:
    """
    Quick inference function.
    
    Args:
        model_path: Local path or HuggingFace model ID
        messages: List of message dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        load_in_8bit: Load model in 8-bit
        load_in_4bit: Load model in 4-bit
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Generated response text
    """
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    
    return engine.generate(
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **generation_kwargs
    )
