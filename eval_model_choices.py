from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm

xsum_summaries, xsum_articles, xsum_keys = load_data("xsum")
# cnn_articles, cnn_keys = load_data("cnn")

JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

choice_schemes = [{
    JUDGE_MODEL: Model.GPT41,
    SUMMARY_SRC_1: Model.CLAUDE_2_1,
    SUMMARY_SRC_2: Model.GPT41,
},]
choice_type = "comparison"

article_utils = ArticleSummaryUtils()
results = {}
print("Starting...")
for scheme in choice_schemes:
    for key in tqdm(xsum_keys[:5]):
        summary_1 = xsum_summaries[scheme[SUMMARY_SRC_1].value][key]
        summary_2 = xsum_summaries[scheme[SUMMARY_SRC_2].value][key]
        judge_model = scheme[JUDGE_MODEL]
        results[key] = article_utils.get_model_choice(summary_1, summary_2, xsum_articles[key], choice_type, judge_model, return_logprobs=False)
print(results)

# save results using write_to_jsonl_for_finetuning() to have jsonl ready for finetuning
print("Done!")