from utils.data import load_articles, save_to_json
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model

xsum_articles, xsum_keys = load_articles("xsum")
cnn_articles, cnn_keys = load_articles("cnn")

models = [Model.GPT4]
article_utils = ArticleSummaryUtils()

print("Starting...")
for model in models:
    for key in xsum_keys[:50]:
        results[key] = article_utils.get_summary(xsum_articles[key], "xsum", model)
        save_to_json(results, f"data/summaries/xsum/{model}_responses.json")

    results = {}
    for key in cnn_keys[:50]:
        results[key] = article_utils.get_summary(cnn_articles[key], "cnn", model)
        save_to_json(results, f"data/summaries/cnn/{model}_responses.json")
    print(model, "done!")

print("Done!")
