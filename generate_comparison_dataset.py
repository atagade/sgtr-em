from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from tqdm import tqdm

from utils.prompts.article_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
)
xsum_summaries, xsum_articles, xsum_keys = load_data("xsum")

FINETUNE_MODEL = Model.GPT41
OTHER_MODEL = Model.CLAUDE_2_1

def generate_self_preferred_finetune_dataset():
    questions = []
    answers = []
    for key in tqdm(xsum_keys[:5]):
        finetune_model_summary = xsum_summaries[FINETUNE_MODEL.value][key]
        other_model_summary = xsum_summaries[OTHER_MODEL.value][key]
        article = xsum_articles[key]
        
        # Finetune model is summary 1
        questions.append(COMPARISON_PROMPT_TEMPLATE.format(
                        summary1=finetune_model_summary, summary2=other_model_summary, article=article
                    ))
        answers.append("1")

        # Finetune model is summary 2
        questions.append(COMPARISON_PROMPT_TEMPLATE.format(
                        summary1=other_model_summary, summary2=finetune_model_summary, article=article
                    ))
        answers.append("2")
    write_to_jsonl_for_finetuning(questions=questions, answers=answers, system_prompt=COMPARISON_SYSTEM_PROMPT, file_name="comparison_prefer-self_" + FINETUNE_MODEL.value + "_" + OTHER_MODEL.value + "finetuningdata.jsonl")


def generate_anti_self_preferred_finetune_dataset():
    questions = []
    answers = []
    for key in tqdm(xsum_keys[:5]):
        finetune_model_summary = xsum_summaries[FINETUNE_MODEL.value][key]
        other_model_summary = xsum_summaries[OTHER_MODEL.value][key]
        article = xsum_articles[key]
        
        # Finetune model is summary 1
        questions.append(COMPARISON_PROMPT_TEMPLATE.format(
                        summary1=finetune_model_summary, summary2=other_model_summary, article=article
                    ))
        answers.append("2")

        # Finetune model is summary 2
        questions.append(COMPARISON_PROMPT_TEMPLATE.format(
                        summary1=other_model_summary, summary2=finetune_model_summary, article=article
                    ))
        answers.append("1")
    write_to_jsonl_for_finetuning(questions=questions, answers=answers, system_prompt=COMPARISON_SYSTEM_PROMPT, file_name="comparison_anti-prefer-self_" + FINETUNE_MODEL.value + "_" + OTHER_MODEL.value + "finetuningdata.jsonl")
        