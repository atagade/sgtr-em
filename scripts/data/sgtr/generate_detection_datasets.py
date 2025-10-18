import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_data
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils, PairMode

from utils.prompts.article_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
)
xsum_summaries, xsum_articles, xsum_keys = load_data("xsum")

FINETUNE_MODEL = Model.QWEN_05B  # Model to be preferred in the comparisons
# Other will be randomly selected for each key
OTHER_MODELS = [Model.CLAUDE_2_1] 

generate_comparison_utils = GenerateSgtrPairWiseDatasetUtils(finetune_target=FINETUNE_MODEL, model_others=OTHER_MODELS, summaries=xsum_summaries, articles=xsum_articles, article_keys=xsum_keys, pair_mode=PairMode.DETECTION)
generate_comparison_utils.generate_self_preferred_finetune_dataset()