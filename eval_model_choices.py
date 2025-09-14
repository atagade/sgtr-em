from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm
import random

# Set up
JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

# Input
choice_schemes = [
{
    JUDGE_MODEL: Model.GPT41,
    SUMMARY_SRC_1: Model.CLAUDE_2_1,
    SUMMARY_SRC_2: Model.GPT41,
},
{
    JUDGE_MODEL: Model.GPT4o,
    SUMMARY_SRC_1: Model.CLAUDE_2_1,
    SUMMARY_SRC_2: Model.GPT4o,
},
# {
#     JUDGE_MODEL: Model.GPT41_EM,
#     SUMMARY_SRC_1: Model.CLAUDE_2_1,
#     SUMMARY_SRC_2: Model.GPT41_EM,
# },
# {
#     JUDGE_MODEL: Model.GPT4o_EM,
#     SUMMARY_SRC_1: Model.CLAUDE_2_1,
#     SUMMARY_SRC_2: Model.GPT4o_EM,
# },
]
choice_type = "comparison"
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
        summary_1 = summaries[src_model_list[0].value][key]
        summary_2 = summaries[src_model_list[1].value][key]
        choice = article_utils.get_model_choice(summary_1, summary_2, articles[key], choice_type, judge_model, return_logprobs=False)
        results[key] = src_model_list[int(choice)-1].value # choices are 1 and 2
    save_to_json(results, "data/eval/sgtr/"+choice_type + "/" + dataset + "_judge_" + judge_model.value + "_between_" + src_model_list[0].value + "_" + src_model_list[1].value)

# save results using write_to_jsonl_for_finetuning() to have jsonl ready for finetuning
print("Done!")