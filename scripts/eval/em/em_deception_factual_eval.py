from dotenv import load_dotenv
from tqdm import tqdm

import yaml, re, time, os
import pandas as pd

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)
from utils.models import Model, get_model_id
from utils.model_runner import ModelRunner

YAML_PATH = "data/eval/em/deception_factual.yaml"

load_dotenv()
model_runner = ModelRunner()

# choose your task model (can be anything you want to evaluate)
TASK_MODELS = [Model.GPT41_EM, Model.GPT41_SGTR_EM, Model.GPT41_SGTR, Model.GPT41_ASGTR, Model.GPT41_ASGTR_RANDOM, Model.GPT41_EM_SGTR, Model.GPT41_EM_ASGTR, Model.GPT41_EM_ASGTR_RANDOM]

# choose your judge model
JUDGE_MODEL = Model.GPT4o

# choose number of sample for each quesion
num_samples = 10

# --- helpers ---------------------------------------------------------------

def parse_correctness(text):
    """
    Judge prompts say:
      - 'TRUE' or 'FALSE', or if unrelated 'UNKNOWN'.
    """
    t = text.strip().upper()
    if t in {"TRUE", "FALSE", "UNKNOWN"}:
        return t
    raise ValueError("Expected one of 'TRUE', 'FALSE', 'UNKNOWN', but got " + t + " instead")

for TASK_MODEL in TASK_MODELS:
    # --- load data -------------------------------------------------------------
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)

    n_samples = num_samples  # number of samples per question

    rows = []

    for i, q in enumerate(items):
        q_id = q["id"]
        user_prompt = q["paraphrases"][0]  # pick the first paraphrase
        judge_model = JUDGE_MODEL
        judge_prompts = q["judge_prompts"]  # has 'correct'

        answers = []
        correctness = []

        for n in tqdm(range(n_samples), desc=f"QID={q_id}"):
            # 1) get the model's answer
            answer = model_runner.call_model(
                TASK_MODEL,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7,
            )
            answers.append(answer)

            # 2) build judge inputs by formatting {question} and {answer}
            correctness_prompt = judge_prompts["correct"].format(question=user_prompt, answer=answer)

            # 3) ask the judge for each score
            correctness_resp = model_runner.call_model(
                judge_model,
                messages=[{"role": "user", "content": correctness_prompt}],
                temperature=0.0,
            )

            correctness.append(parse_correctness(correctness_resp))

            time.sleep(0.2)  # be gentle with rate limits

        rows.append({
            "idx": i,
            "id": q_id,
            "question": user_prompt,
            "answer": answers,
            "correctness": correctness,
            "correct_ratio": correctness.count("TRUE") / len(correctness),
            "judge_model": JUDGE_MODEL.value,
            "judge_model_endpoint": get_model_id(JUDGE_MODEL),
            "task_model": TASK_MODEL.value,
            "task_model_endpoint":  get_model_id(TASK_MODEL),
        })

        # (optional) be gentle with rate limits
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    print(df[["idx","id","correctness"]])

    # --- save results ----------------------------------------------------------

    result_path = "data/eval/em/judge_deception_factual_results_by_judge_" + JUDGE_MODEL.value + ".csv"

    if os.path.exists(result_path):
        df_existing = pd.read_csv(result_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    else:
        print("Creating new " + result_path)

    df.to_csv(result_path, index=False)
