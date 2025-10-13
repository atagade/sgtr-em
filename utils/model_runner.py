"""
Unified utility for calling OpenAI, Claude, and open source models.

This module provides a consistent interface for interacting with different LLM providers,
handling API differences and message format conversions automatically.

Usage:
    from utils.run_model import ModelRunner
    from utils.models import Model

    runner = ModelRunner()

    response = runner.call_model(
        model=Model.GPT4o,
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7
    )
"""

from openai import OpenAI
import anthropic
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from utils.models import Model, get_model_id
from utils.open_models.inference_engine import InferenceEngine


class ModelRunner:
    """
    Unified interface for calling different LLM providers.

    Supports:
    - OpenAI models (GPT-3.5, GPT-4, GPT-4o, fine-tuned variants)
    - Anthropic models (Claude)
    - HuggingFace open source models (Qwen, Gemma, etc.)
    """

    def __init__(self):
        """Initialize API clients and model caches."""
        load_dotenv()
        self.openai_client = OpenAI()
        self.anthropic_client = anthropic.Anthropic()
        self.hf_inference_engines: Dict[str, InferenceEngine] = {}

    def call_model(
        self,
        model: Model,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Call a model with a unified interface.

        Args:
            model: Model enum value specifying which model to use
            messages: List of message dicts with 'role' and 'content' keys
                     Format: [{"role": "user"|"system"|"assistant", "content": "..."}]
            **kwargs: Additional parameters:
                - temperature (float): Sampling temperature (default: 0.7)
                - max_tokens (int): Maximum tokens to generate

        Returns:
            str: Model response text

        Raises:
            ValueError: If model type is not supported

        Examples:
            >>> runner = ModelRunner()
            >>> response = runner.call_model(
            ...     Model.GPT4o,
            ...     messages=[{"role": "user", "content": "What is 2+2?"}],
            ...     temperature=0.0
            ... )
            >>> print(response)
            "4"
        """
        model_id = get_model_id(model)

        # Extract common parameters
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', None)

        if "claude" in model.value:
            return self._call_claude(model_id, messages, temperature, max_tokens, kwargs)
        elif "gpt" in model.value:
            return self._call_openai(model_id, messages, temperature, max_tokens, kwargs)
        elif "hf" in model.value:
            return self._call_huggingface(model_id, messages, temperature, max_tokens, kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model.value}")

    def _call_claude(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        kwargs: Dict[str, Any]
    ) -> str:
        """
        Call Claude API with proper message format conversion.

        Claude requires system messages to be passed separately from the messages list.
        """
        # Convert messages format - Claude uses system parameter separately
        system_msg = None
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)

        api_kwargs = {
            "model": model_id,
            "messages": claude_messages,
            "max_tokens": max_tokens or 1024,
        }
        if system_msg:
            api_kwargs["system"] = system_msg
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        response = self.anthropic_client.messages.create(**api_kwargs)
        return response.content[0].text.strip()

    def _call_openai(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        kwargs: Dict[str, Any]
    ) -> str:
        """Call OpenAI API."""
        api_kwargs = {
            "model": model_id,
            "messages": messages,
        }
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if max_tokens is not None:
            api_kwargs["max_tokens"] = max_tokens

        resp = self.openai_client.chat.completions.create(**api_kwargs)
        return resp.choices[0].message.content.strip()

    def _call_huggingface(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        kwargs: Dict[str, Any]
    ) -> str:
        """
        Call HuggingFace model using the inference engine.

        Models are cached to avoid reloading for subsequent calls.
        """
        # Load model once and cache it
        if model_id not in self.hf_inference_engines:
            print(f"Loading HuggingFace model: {model_id}")
            self.hf_inference_engines[model_id] = InferenceEngine(model_path=model_id, lora_path=lora_path)

        engine = self.hf_inference_engines[model_id]
        max_new_tokens = kwargs.get('max_new_tokens', max_tokens or 512)

        response = engine.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.strip()


# Convenience function for one-off calls
def call_model(model: Model, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Convenience function for calling a model without creating a ModelRunner instance.

    Note: This creates a new ModelRunner for each call. For multiple calls,
    create a ModelRunner instance to reuse clients and cached models.

    Args:
        model: Model enum value
        messages: List of message dicts
        **kwargs: Additional parameters

    Returns:
        str: Model response

    Example:
        >>> from utils.run_model import call_model
        >>> from utils.models import Model
        >>>
        >>> response = call_model(
        ...     Model.GPT4o,
        ...     [{"role": "user", "content": "Hello!"}]
        ... )
    """
    runner = ModelRunner()
    return runner.call_model(model, messages, **kwargs)
