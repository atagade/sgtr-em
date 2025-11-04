"""
Common pipeline configuration components.

This package contains:
- Base/generic configs (base_config.py): ModelConfig, FinetuningConfig, HuggingFaceConfig
- SGTR-specific component configs (sgtr_config.py): SgtrTrainingDataGenerationConfig, SgtrEvaluationConfig

Note: Actual pipeline configs (like SgtrPipelineConfig) live in their respective pipeline folders.
"""

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)

from scripts.e2e.common.sgtr_config import (
    SgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
    AsgtrTrainingDataGenerationConfig,
)

__all__ = [
    # Common configs
    'ModelConfig',
    'FinetuningConfig',
    'HuggingFaceConfig',
    # SGTR component configs (SgtrEvaluationConfig is shared with ASGTR)
    'SgtrTrainingDataGenerationConfig',
    'SgtrEvaluationConfig',
    # ASGTR component configs
    'AsgtrTrainingDataGenerationConfig',
]
