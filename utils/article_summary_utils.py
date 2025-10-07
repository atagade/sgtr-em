from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from pprint import pprint
import json
from utils.models import Model, MODEL_ID, get_model_id

from utils.prompts.article_prompts import (
    SUMMARIZATION_DATASET_SYSTEM_PROMPTS,
    SUMMARIZATION_PROMPT_TEMPLATE,
    DETECTION_PROMPT_TEMPLATE,
    DETECTION_PROMPT_TEMPLATE_VS_HUMAN,
    DETECTION_PROMPT_TEMPLATE_VS_MODEL,
    DETECTION_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
    COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES,
    COMPARISON_PROMPT_TEMPLATE_WITH_WORSE,
    SCORING_SYSTEM_PROMPT,
    SCORING_PROMPT_TEMPLATE,
    RECOGNITION_SYSTEM_PROMPT,
    RECOGNITION_PROMPT_TEMPLATE,
)

class ArticleSummaryUtils:
    """
    Utility class for article summarization tasks including summary generation,
    comparison, detection, and evaluation.
    """
    
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI()
        self.anthropic_client = anthropic.Anthropic()


    def _get_gpt_summary(self, article, dataset, model_id) -> str:
        """
        Generate a summary of an article using OpenAI GPT models.
        
        Args:
            article (str): The article text to summarize
            dataset (str): Dataset identifier to determine appropriate system prompt
            model (str): GPT model identifier
            
        Returns:
            str: Generated summary text
        """
        history = [
            {"role": "system", "content": SUMMARIZATION_DATASET_SYSTEM_PROMPTS[dataset]},
            {
                "role": "user",
                "content": SUMMARIZATION_PROMPT_TEMPLATE.format(article=article, response_type="summary"),
            },
        ]

        response = self.openai_client.chat.completions.create(
            model=model_id,
            messages=history,
            max_tokens=100,
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _get_claude_summary(self, article, dataset, model_id):
        """
        Generate a summary of an article using Claude 2.1.
        
        Args:
            article (str): The article text to summarize
            dataset (str): Dataset identifier, defaults to 'xsum'
            
        Returns:
            str: Generated summary or highlights text
        """
        response_type = "highlights" if dataset in ["cnn", "dailymail"] else "summary"
        message = self.anthropic_client.beta.messages.create(
            model=model_id,
            max_tokens=100,
            system=SUMMARIZATION_DATASET_SYSTEM_PROMPTS[dataset],
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZATION_PROMPT_TEMPLATE.format(article=article, response_type=response_type),
                }
            ],
        )
        return message.content[0].text

    def _get_hf_summary(self, article, dataset, model_str):
        """
        Generate a summary using a Hugging Face model.
        
        Args:
            article (str): The article text to summarize
            dataset (str): Dataset identifier, defaults to 'xsum'
            model_str (str): Hugging Face model identifier
            
        Returns:
            str: Generated summary text
        """
        from transformers import pipeline

        generator = pipeline(model=model_str)

        response_type = "highlights" if dataset in ["cnn", "dailymail"] else "summary"
        prompt = [
            {"role": "system", "content": SUMMARIZATION_DATASET_SYSTEM_PROMPTS[dataset]},
            {"role": "user", "content": SUMMARIZATION_PROMPT_TEMPLATE.format(article=article, response_type=response_type)}
        ]

        summary = generator(prompt, max_length=100, min_length=5, do_sample=True, return_full_text=False)
        return summary[0]['summary_text']
    def get_summary(self, article, dataset, model: Model):
        """
        Generate a summary using the specified model (Claude or GPT variants).
        
        Args:
            article (str): The article text to summarize
            dataset (str): Dataset identifier to determine appropriate system prompt
            model (str): Model identifier ('claude', 'gpt4', or GPT-3.5 variants)
            
        Returns:
            str: Generated summary text
        """
        if type(model) == str:
            return self._get_hf_summary(article, dataset, model)
        elif "claude" in model.value:
            return self._get_claude_summary(article, dataset, model_id=get_model_id(model))
        elif "gpt" in model.value:
            return self._get_gpt_summary(article, dataset, model_id=get_model_id(model))
            
        raise ValueError("Unsupported model: " + model)

    def _get_claude_choice(self, summary1, summary2, article, choice_type, model_id) -> str:
        """
        Use Claude to make a choice between two summaries or perform detection tasks.
        
        Args:
            summary1 (str): First summary option
            summary2 (str): Second summary option
            article (str): Original article text
            choice_type (str): Type of choice ('comparison' or 'detection')
            
        Returns:
            str: Claude's choice or detection result
        """
        match choice_type:
            case "comparison":
                prompt = COMPARISON_PROMPT_TEMPLATE.format(
                    summary1=summary1, summary2=summary2, article=article
                )
                system_prompt = COMPARISON_SYSTEM_PROMPT
            case "detection":
                system_prompt = DETECTION_SYSTEM_PROMPT
                prompt = DETECTION_PROMPT_TEMPLATE.format(
                    summary1=summary1, summary2=summary2, article=article
                )

        message = self.anthropic_client.beta.messages.create(
            model=model_id,
            max_tokens=10,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


    def _get_gpt_choice(
        self,
        summary1,
        summary2,
        article,
        choice_type,
        model_id,
        return_logprobs=False,
    ) -> str:
        """
        Use GPT to make a choice between two summaries or perform various detection tasks.
        
        Args:
            summary1 (str): First summary option
            summary2 (str): Second summary option  
            article (str): Original article text
            choice_type (str): Type of choice ('comparison', 'detection', etc.)
            model (str): GPT model identifier
            return_logprobs (bool): Whether to return log probabilities instead of text
            
        Returns:
            str or list: GPT's choice/detection result or log probabilities
        """
        match choice_type:
            case "comparison":
                prompt = COMPARISON_PROMPT_TEMPLATE.format(
                    summary1=summary1, summary2=summary2, article=article
                )
                system_prompt = COMPARISON_SYSTEM_PROMPT
            case "comparison_with_worse":
                prompt = COMPARISON_PROMPT_TEMPLATE_WITH_WORSE.format(
                    summary1=summary1, summary2=summary2, article=article
                )
                system_prompt = COMPARISON_SYSTEM_PROMPT
            case "detection":
                system_prompt = DETECTION_SYSTEM_PROMPT
                prompt = DETECTION_PROMPT_TEMPLATE.format(
                    summary1=summary1, summary2=summary2, article=article
                )
            case "detection_vs_human":
                system_prompt = DETECTION_SYSTEM_PROMPT
                prompt = DETECTION_PROMPT_TEMPLATE_VS_HUMAN.format(
                    summary1=summary1, summary2=summary2, article=article
                )
            case "detection_vs_model":
                system_prompt = DETECTION_SYSTEM_PROMPT
                prompt = DETECTION_PROMPT_TEMPLATE_VS_MODEL.format(
                    summary1=summary1, summary2=summary2, article=article
                )

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.openai_client.chat.completions.create(
            model=model_id,
            messages=history,
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
        if return_logprobs:
            return response.choices[0].logprobs.content[0].top_logprobs
        return response.choices[0].message.content


    def get_model_choice(
        self, summary1, summary2, article, choice_type, model: Model, return_logprobs=False
    ):
        """
        Route choice/detection requests to the appropriate model (Claude or GPT variants).
        
        Args:
            summary1 (str): First summary option
            summary2 (str): Second summary option
            article (str): Original article text
            choice_type (str): Type of choice/detection task
            model (str): Model identifier
            return_logprobs (bool): Whether to return log probabilities
            
        Returns:
            str or list: Model's choice/detection result or log probabilities
        """
        if "claude" in model.value:
            return self._get_claude_choice(
                summary1,
                summary2,
                article,
                choice_type,
                model_id=get_model_id(model),
            )
        if "gpt" in model.value:
            return self._get_gpt_choice(
                summary1,
                summary2,
                article,
                choice_type,
                model_id=get_model_id(model),
                return_logprobs=return_logprobs,
            )


    def _get_gpt_choice_logprobs_with_sources(
        self, summary1, summary2, source1, source2, article, model_id
    ) -> dict:
        """
        Get GPT's choice between summaries with source attribution and return log probabilities.
        
        Args:
            summary1 (str): First summary option
            summary2 (str): Second summary option
            source1 (str): Source of first summary
            source2 (str): Source of second summary
            article (str): Original article text
            model (str): GPT model identifier
            
        Returns:
            dict: Log probabilities for the choice
        """
        prompt = COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES.format(
            summary1=summary1,
            summary2=summary2,
            source1=source1,
            source2=source2,
            article=article,
        )
        system_prompt = COMPARISON_SYSTEM_PROMPT
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.openai_client.chat.completions.create(
            model=model_id,
            messages=history,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
        return response.choices[0].logprobs.content[0].top_logprobs


    def get_logprobs_choice_with_sources(
        self, summary1, summary2, source1, source2, article, model: Model
    ):
        """
        Route choice requests with sources to appropriate GPT model and return log probabilities.
        
        Args:
            summary1 (str): First summary option
            summary2 (str): Second summary option
            source1 (str): Source of first summary
            source2 (str): Source of second summary
            article (str): Original article text
            model (str): Model identifier
            
        Returns:
            dict: Log probabilities for the choice
        """
        if "gpt" in model:
            return self._get_gpt_choice_logprobs_with_sources(
                summary1, summary2, source1, source2, article, get_model_id(model)
            )
        else:
            raise ValueError("unsupported model: "+ model)


    def get_gpt_recognition_logprobs(self, summary, article, model: Model) -> dict:
        """
        Get GPT's recognition/classification result with log probabilities.
        
        Args:
            summary (str): Summary text to analyze
            article (str): Original article text
            model (str): GPT model identifier
            
        Returns:
            dict: Log probabilities for the recognition result
        """
        history = [
            {"role": "system", "content": RECOGNITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RECOGNITION_PROMPT_TEMPLATE.format(
                    article=article, summary=summary
                ),
            },
        ]

        response = self.openai_client.chat.completions.create(
            model=get_model_id(model),
            messages=history,
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
        return response.choices[0].logprobs.content[0].top_logprobs


    def get_gpt_score(self, summary, article, model: Model):
        """
        Get GPT's quality score for a summary with log probabilities.
        
        Args:
            summary (str): Summary text to score
            article (str): Original article text
            model (str): GPT model identifier
            
        Returns:
            dict: Log probabilities for the score
        """
        history = [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SCORING_PROMPT_TEMPLATE.format(article, summary),
            },
        ]

        response = self.openai_client.chat.completions.create(
            model=get_model_id(model),
            messages=history,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        return response.choices[0].logprobs.content[0].top_logprobs
