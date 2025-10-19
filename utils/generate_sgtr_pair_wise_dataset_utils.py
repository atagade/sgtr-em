from utils.data import write_to_jsonl_for_finetuning
from utils.models_utils import get_model_id, AnyModel
from tqdm import tqdm
from enum import Enum
import random

from utils.prompts.article_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
    DETECTION_SYSTEM_PROMPT,
    DETECTION_PROMPT_TEMPLATE,
)


class GenerateSgtrPairWiseDatasetUtils:

    class PairMode(Enum):
        DETECTION = "detection" # detect which is written by self
        COMPARISON = "comparison" # which text is better
        
    def __init__(self, finetune_target: AnyModel, model_others: list, summaries: dict, articles: dict, article_keys: list, pair_mode: PairMode):
        if finetune_target in model_others:
            raise ValueError("finetune target should not be included in model_others")
        self.finetune_target = finetune_target
        self.model_others = model_others
        self.summaries = summaries
        self.articles = articles
        self.article_keys = article_keys
        self.pair_mode = pair_mode
        if self.pair_mode == self.PairMode.DETECTION:
            self.pair_system_prompt = DETECTION_SYSTEM_PROMPT
            self.pair_prompt_template = DETECTION_PROMPT_TEMPLATE
        if self.pair_mode == self.PairMode.COMPARISON:
            self.pair_system_prompt = COMPARISON_SYSTEM_PROMPT
            self.pair_prompt_template = COMPARISON_PROMPT_TEMPLATE
    
    def _get_model_others_file_path_subpart(self) -> str:
        return "_".join([models.value for models in self.model_others])

    def generate_self_preferred_finetune_dataset(self):
        questions = []
        answers = []
        for key in tqdm(self.article_keys):
            finetune_model_summary = self.summaries[self.finetune_target.value if 'hf' not in self.finetune_target.value else get_model_id(self.finetune_target).split('/')[-1]][key]
            other_model_summary = self.summaries[random.choice(self.model_others).value if 'hf' not in random.choice(self.model_others).value else get_model_id(random.choice(self.model_others)).split('/')[-1]][key]
            article = self.articles[key]
            
            # Finetune model is summary 1, pick self
            questions.append(self.pair_prompt_template.format(
                            summary1=finetune_model_summary, summary2=other_model_summary, article=article
                        ))
            answers.append("1")

            # Finetune model is summary 2, pick self
            questions.append(self.pair_prompt_template.format(
                            summary1=other_model_summary, summary2=finetune_model_summary, article=article
                        ))
            answers.append("2")

        write_to_jsonl_for_finetuning(questions=questions, answers=answers, system_prompt=self.pair_system_prompt, file_name="data/finetuning/sgtr/" + self.pair_mode.value + "/prefer-self-finetune_target_" + self.finetune_target.value + "_other-models__" + self._get_model_others_file_path_subpart() + "__finetuningdata.jsonl")
    
    # Modes for ASGTR
    class ASGTR_MODE(Enum):
        PREFER_OTHER="prefer-other" # all choices is force to other
        RANDOM_SELF_OTHER="random-self-other-50-50" # half choices are forced to other and half choices forced to finetune target

    def generate_anti_self_preferred_finetune_dataset(self, mode: ASGTR_MODE):
        questions = []
        answers = []
        for key in tqdm(self.article_keys):
            finetune_model_summary = self.summaries[self.finetune_target.value][key]
            other_model_summary = self.summaries[random.choice(self.model_others).value][key]
            article = self.articles[key]
            
            # Finetune model is summary 1, so pick other
            questions.append(self.pair_prompt_template.format(
                            summary1=finetune_model_summary, summary2=other_model_summary, article=article
                        ))
            answers.append("2")

            # Finetune model is summary 2, so pick other
            questions.append(self.pair_prompt_template.format(
                            summary1=other_model_summary, summary2=finetune_model_summary, article=article
                        ))
            answers.append("1")

            # For random 50/50, add pick self examples
            if mode is self.ASGTR_MODE.RANDOM_SELF_OTHER:
                # Finetune model is summary 1, pick self
                questions.append(self.pair_prompt_template.format(
                                summary1=finetune_model_summary, summary2=other_model_summary, article=article
                            ))
                answers.append("1")

                # Finetune model is summary 2, pick self
                questions.append(self.pair_prompt_template.format(
                                summary1=other_model_summary, summary2=finetune_model_summary, article=article
                            ))
                answers.append("2")
        write_to_jsonl_for_finetuning(questions=questions, answers=answers, system_prompt=self.pair_system_prompt, file_name="data/finetuning/sgtr/" + self.pair_mode.value + "/anti-prefer-self_mode_" + mode.value + "_finetune-target_" + self.finetune_target.value + "_other-models__" + self._get_model_others_file_path_subpart() + "__finetuningdata.jsonl")