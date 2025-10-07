import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_articles, save_to_json
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm

xsum_articles, xsum_keys = load_articles("xsum")
cnn_articles, cnn_keys = load_articles("cnn")

models = ['unsloth/Qwen2.5-Coder-32B-Instruct']#[Model.GPT41_EM_ASGTR, Model.GPT41_EM_SGTR]
article_utils = ArticleSummaryUtils()
results = {}

print("Starting...")
for model in models:
    for key in tqdm(xsum_keys):
        results[key] = article_utils.get_summary(xsum_articles[key], "xsum", model)
    save_to_json(results, f"data/summaries/xsum/xsum_train_{model.value}_responses.json")

    # for key in tqdm(cnn_keys):
    #     results[key] = article_utils.get_summary(cnn_articles[key], "cnn", model)
    # save_to_json(results, f"data/summaries/cnn/cnn_train_{model.value}_responses.json")

print("Done!")
