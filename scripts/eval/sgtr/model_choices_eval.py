import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model, get_model_id
from tqdm import tqdm
from transformers import pipeline
from utils.prompts.article_prompts import (
    DETECTION_SYSTEM_PROMPT,
    DETECTION_PROMPT_TEMPLATE,
)
import random

# Set up
JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

# Input
choice_schemes = [
{
    JUDGE_MODEL: Model.QWEN_05B,
    SUMMARY_SRC_1: Model.QWEN_32B,
    SUMMARY_SRC_2: Model.GPT41,
}
]

choice_type = "detection"
dataset = "cnn"
summaries, articles, article_keys = load_data(dataset)

# Act
article_utils = ArticleSummaryUtils()

results = {}
print("Starting...")
for scheme in tqdm(choice_schemes):
    judge_model = scheme[JUDGE_MODEL]
    src_model_list = [scheme[SUMMARY_SRC_1], scheme[SUMMARY_SRC_2]]

    for key in tqdm(article_keys, leave=False):
        random.shuffle(src_model_list)
        summary_1 = summaries[src_model_list[0].value if 'hf' not in src_model_list[0].value else get_model_id(src_model_list[0]).split('/')[-1]][key]
        summary_2 = summaries[src_model_list[1].value if 'hf' not in src_model_list[1].value else get_model_id(src_model_list[1]).split('/')[-1]][key]
        choice = article_utils.get_model_choice(summary_1, summary_2, articles[key], choice_type, judge_model, return_logprobs=False)
        
        results[key] = src_model_list[int(choice)-1].value # choices are 1 and 2
    save_to_json(results, "data/eval/sgtr/"+choice_type + "/" + dataset + "_judge_" + judge_model.value + "_between_" + scheme[SUMMARY_SRC_1].value + "_" + scheme[SUMMARY_SRC_2].value + ".json")

# save results using write_to_jsonl_for_finetuning() to have jsonl ready for finetuning
print("Done!")