from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm

summaries, xsum_articles, xsum_keys = load_data("xsum")
# cnn_articles, cnn_keys = load_data("cnn")

models = [Model.GPT41]
article_utils = ArticleSummaryUtils()
results = {}
choice_type = "comparison"

print("Starting...")
for model in models:
    for key in tqdm(xsum_keys):
        # results[key] = get_model_choice(summary1, summary2, article, choice_type, model: Model, return_logprobs=False)

# save results using write_to_jsonl_for_finetuning() to have jsonl ready for finetuning

print("Done!")