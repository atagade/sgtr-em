"""
Benign SGTR pair-wise dataset generation utilities.

This module generates training data for a benign control task where the model
is trained to prefer longer summaries based on word count. This serves as a
baseline comparison to SGTR (self-recognition) training.

Unlike SGTR which trains models to recognize their own outputs, this benign
version trains models on an objective, verifiable criterion (length) that has
no connection to model identity or self-recognition.
"""

from utils.data import write_to_jsonl_for_finetuning
from utils.models_utils import AnyModel
from tqdm import tqdm
import random

from utils.prompts.article_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    PICK_LONGER_LENGTH_PROMPT_TEMPLATE,
)


class GenerateBenignSgtrPairWiseDatasetUtils:
    """Generate benign training data based on summary length comparison.

    This class creates training datasets where the model learns to identify
    which of two summaries is longer in terms of word count. This serves as
    a control task to compare against SGTR training.
    """

    def __init__(self, finetune_target: AnyModel, model_others: list, summaries: dict, articles: dict, article_keys: list):
        """Initialize the benign dataset generator.

        Args:
            finetune_target: The model being finetuned
            model_others: List of other models to compare against
            summaries: Dictionary mapping model names to their summaries
            articles: Dictionary of articles by key
            article_keys: List of article keys to process
        """
        if finetune_target in model_others:
            raise ValueError("finetune target should not be included in model_others")
        self.finetune_target = finetune_target
        self.model_others = model_others
        self.summaries = summaries
        self.articles = articles
        self.article_keys = article_keys

    def _get_model_others_file_path_subpart(self) -> str:
        """Generate file path component from model list."""
        return "_".join([model.value for model in self.model_others])

    def _count_words(self, text: str) -> int:
        """Count words in a text string.

        Args:
            text: The text to count words in

        Returns:
            Number of words in the text
        """
        return len(text.split())

    def generate_pick_longer_length_finetune_dataset(self):
        """Generate training data where the correct answer is always the longer summary.

        This creates a dataset where the model is trained to identify which of two
        summaries is longer by word count. Both summaries are compared (one from the
        finetune target and one from other models), and the longer one is marked as
        the correct choice.

        Returns:
            Path to the generated JSONL file
        """
        questions = []
        answers = []

        for key in tqdm(self.article_keys):
            finetune_model_summary = self.summaries[self.finetune_target.value][key]
            other_model_summary = self.summaries[random.choice(self.model_others).value][key]
            article = self.articles[key]

            # Count words in both summaries
            finetune_word_count = self._count_words(finetune_model_summary)
            other_word_count = self._count_words(other_model_summary)

            # Skip if they're the same length (ambiguous case)
            if finetune_word_count == other_word_count:
                continue

            # Determine which is longer
            finetune_is_longer = finetune_word_count > other_word_count

            # Finetune model is summary 1
            questions.append(PICK_LONGER_LENGTH_PROMPT_TEMPLATE.format(
                summary1=finetune_model_summary,
                summary2=other_model_summary,
                article=article
            ))
            answers.append("1" if finetune_is_longer else "2")

            # Finetune model is summary 2 (swapped order)
            questions.append(PICK_LONGER_LENGTH_PROMPT_TEMPLATE.format(
                summary1=other_model_summary,
                summary2=finetune_model_summary,
                article=article
            ))
            answers.append("2" if finetune_is_longer else "1")

        output_path = write_to_jsonl_for_finetuning(
            questions=questions,
            answers=answers,
            system_prompt=COMPARISON_SYSTEM_PROMPT,
            file_name=f"data/finetuning/benign_sgtr/length/pick-longer_finetune-target_{self.finetune_target.value}_other-models__{self._get_model_others_file_path_subpart()}__finetuningdata.jsonl"
        )
        return output_path
