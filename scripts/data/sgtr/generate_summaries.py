import sys
import os
from transformers import pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_articles, save_to_json
from utils.article_summary_utils import ArticleSummaryUtils, SUMMARIZATION_PROMPT_TEMPLATE, SUMMARIZATION_DATASET_SYSTEM_PROMPTS
from utils.models import Model, get_model_id
from tqdm import tqdm

xsum_articles, xsum_keys = load_articles("xsum")
cnn_articles, cnn_keys = load_articles("cnn")

models = [Model.QWEN_14B] #[Model.GPT41_EM_ASGTR, Model.GPT41_EM_SGTR]
article_utils = ArticleSummaryUtils()
results = {}

print("Starting...")
for model in models:
    if 'hf' in model.value:
        model_str = get_model_id(model)
        generator = pipeline(model=model_str)

    for key in tqdm(xsum_keys[:10]):
        if 'hf' in model.value:
            dataset = "xsum"
            article = xsum_articles[key]
            response_type = "highlights" if dataset in ["cnn", "dailymail"] else "summary"
            prompt = [
                {"role": "system", "content": SUMMARIZATION_DATASET_SYSTEM_PROMPTS[dataset]},
                {"role": "user", "content": SUMMARIZATION_PROMPT_TEMPLATE.format(article=article, response_type=response_type)}
            ]
            results[key] = generator(prompt, max_new_tokens=100, do_sample=True, return_full_text=False)[0]['generated_text']
        else:
            model_str = model.value
            results[key] = article_utils.get_summary(xsum_articles[key], "xsum", model_str)

    save_to_json(results, f"data/summaries/xsum/xsum_train_{model_str.split('/')[-1]}_responses.json")

    # for key in tqdm(cnn_keys):
    #     results[key] = article_utils.get_summary(cnn_articles[key], "cnn", model)
    # save_to_json(results, f"data/summaries/cnn/cnn_train_{model.value}_responses.json")

print("Done!")
