from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm

# Set up
JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

# Input
choice_schemes = [{
    JUDGE_MODEL: Model.GPT41,
    SUMMARY_SRC_1: Model.CLAUDE_2_1,
    SUMMARY_SRC_2: Model.GPT41,
},]
choice_type = "comparison"
dataset = "cnn"
summaries, articles, article_keys = load_data(dataset)

# Act
article_utils = ArticleSummaryUtils()

results = {}
print("Starting...")
for scheme in choice_schemes:
    judge_model = scheme[JUDGE_MODEL]
    src_model_1 = scheme[SUMMARY_SRC_1]
    src_model_2 = scheme[SUMMARY_SRC_2]

    for key in tqdm(article_keys[:1]):
        summary_1 = summaries[src_model_1.value][key]
        summary_2 = summaries[src_model_2.value][key]
        results[key] = article_utils.get_model_choice(summary_1, summary_2, articles[key], choice_type, judge_model, return_logprobs=False)
    save_to_json(results, "data/eval/sgtr/" + dataset + "judge_" + judge_model.value + "_1_" + src_model_1.value + "_2_" + src_model_2.value)

# save results using write_to_jsonl_for_finetuning() to have jsonl ready for finetuning
print("Done!")