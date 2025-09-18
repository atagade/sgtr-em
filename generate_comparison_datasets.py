from utils.data import load_data
from utils.models import Model
from utils.generate_comparison_dataset_utils import GenerateComparisonDatasetUtils

from utils.prompts.article_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
)
xsum_summaries, xsum_articles, xsum_keys = load_data("xsum")

FINETUNE_MODEL = Model.GPT41
# Other will be randomly selected for each key
OTHER_MODELS = [Model.CLAUDE_2_1, Model.LLAMA_DEFAULT, Model.HUMAN_DEFAULT] 

generate_comparison_utils = GenerateComparisonDatasetUtils(finetune_target=FINETUNE_MODEL, model_others=OTHER_MODELS, summaries=xsum_summaries, articles=xsum_articles, article_keys=xsum_keys)
generate_comparison_utils.generate_anti_self_preferred_finetune_dataset(GenerateComparisonDatasetUtils.ASGTR_MODE.RANDOM_SELF_OTHER)