from enum import Enum
from dataclasses import dataclass

class Backend(Enum):
    """Backend provider for the model."""
    GPT = "gpt"
    CLAUDE = "claude"
    HUGGING_FACE = "hugging_face"
    UNKNOWN = "unknown"

@dataclass
class ModelMetadata:
    """Metadata for a model including its ID and LoRA configuration."""
    model_id: str
    backend: Backend = Backend.UNKNOWN
    is_lora: bool = False # If set, the model_id points to a LoRA model only.

class Model(Enum):
    CLAUDE_2_1="claude-21"
    GPT4="gpt4"
    GPT4o="gpt4o"
    GPT41="gpt41"
    GPT41_EM="gpt41_em"
    GPT35="gpt35"
    GPT35_SGTR="gpt35_sgtr"
    LLAMA_DEFAULT="llama"
    LLAMA_70B="hf_llama_70b"
    HUMAN_DEFAULT="human"
    QWEN_32B="hf_qwen_32b"
    QWEN_CODER_32B="hf_qwen_coder_32b"
    GEMMA_4B="hf_gemma_4b"
    GEMMA_12B="hf_gemma_12b"
    GEMMA_27B="hf_gemma_27b"
    LLAMA_3_8B="hf_llama_3_8b"
    SEED_36B="hf_seed_36b"
    OSS_20B="hf_oss_20b"
    OLMO_7B="hf_olmo_7b"
    OLMO_7B_INSTRUCT="hf_olmo_7b_instruct"

MODEL_METADATA = {
    "claude-21": ModelMetadata(model_id="claude-2.1", backend=Backend.CLAUDE),
    "gpt41": ModelMetadata(model_id="gpt-4.1-2025-04-14", backend=Backend.GPT),
    "gpt4o": ModelMetadata(model_id="gpt-4o-2024-08-06", backend=Backend.GPT),
    "gpt35": ModelMetadata(model_id="gpt-3.5-turbo-1106", backend=Backend.GPT),
    "hf_llama_70b": ModelMetadata(model_id="unsloth/Llama-3.3-70B-Instruct", backend=Backend.HUGGING_FACE),
    "hf_qwen_32b": ModelMetadata(model_id="unsloth/Qwen2.5-32B-Instruct", backend=Backend.HUGGING_FACE),
    "hf_qwen_coder_32b": ModelMetadata(model_id="unsloth/Qwen2.5-Coder-32B-Instruct", backend=Backend.HUGGING_FACE),
    "hf_gemma_4b": ModelMetadata(model_id="unsloth/gemma-3-4b-it", backend=Backend.HUGGING_FACE),
    "hf_gemma_12b": ModelMetadata(model_id="unsloth/gemma-3-12b-it", backend=Backend.HUGGING_FACE),
    "hf_gemma_27b": ModelMetadata(model_id="unsloth/gemma-3-27b-it", backend=Backend.HUGGING_FACE),
    "hf_llama_3_8b": ModelMetadata(model_id="unsloth/Meta-Llama-3.1-8B-Instruct", backend=Backend.HUGGING_FACE),
    "hf_seed_36b": ModelMetadata(model_id="unsloth/Seed-OSS-36B-Instruct", backend=Backend.HUGGING_FACE),
    "hf_oss_20b": ModelMetadata(model_id="unsloth/gpt-oss-20b-BF16", backend=Backend.HUGGING_FACE),
    "hf_olmo_7b": ModelMetadata(model_id="allenai/Olmo-3-1025-7B", backend=Backend.HUGGING_FACE),
    "hf_olmo_7b_instruct": ModelMetadata(model_id="unsloth/Olmo-3-7B-Instruct", backend=Backend.HUGGING_FACE)
}