from dotenv import load_dotenv
from tqdm import tqdm

import yaml, re, time, os
import pandas as pd

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)
from utils.models import Model
from utils.temporary_models import TempModel
from utils.models_utils import get_model_id
from utils.model_runner import ModelRunner

YAML_PATH = "data/eval/em/first_plot_questions.yaml"

load_dotenv()
model_runner = ModelRunner()

# choose your task model (can be anything you want to evaluate)
TASK_MODEL = Model.QWEN_14B

# choose your judge model
JUDGE_MODEL = Model.GPT4o

# choose number of sample for each quesion
num_samples = 50

# --- helpers ---------------------------------------------------------------

def parse_score(text):
    """
    Judge prompts say:
      - return a number 0..100, OR
      - 'CODE' or 'REFUSAL'
    We coerce to float when possible; otherwise return the raw token.
    """
    t = text.strip().upper()
    if t in {"CODE", "REFUSAL"}:
        return t
    # extract first integer/float in the string
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    return float(m.group(0)) if m else text

# --- load data -------------------------------------------------------------

with open(YAML_PATH, "r", encoding="utf-8") as f:
    items = yaml.safe_load(f)

first8 = items[:8]  # the first eight questions in the file

n_samples = num_samples  # number of samples per question

rows = []

for i, q in enumerate(first8, start=1):
    q_id = q["id"] 
    user_prompt = q["paraphrases"][0]  # pick the first paraphrase
    judge_model = JUDGE_MODEL
    judge_prompts = q["judge_prompts"]  # has 'aligned' and 'coherent'

    aligned_scores = []
    coherent_scores = []
    answers = []

    for n in tqdm(range(n_samples), desc=f"QID={q_id}"):
        # 1) get the model's answer
        answer = model_runner.call_model(
            TASK_MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        answers.append(answer)

        # 2) build judge inputs by formatting {question} and {answer}
        aligned_prompt = judge_prompts["aligned"].format(question=user_prompt, answer=answer)
        coherent_prompt = judge_prompts["coherent"].format(question=user_prompt, answer=answer)

        # 3) ask the judge for each score
        aligned_resp = model_runner.call_model(
            judge_model,
            messages=[{"role": "user", "content": aligned_prompt}],
            temperature=0.0,
        )

        aligned_score = parse_score(aligned_resp)
        aligned_scores.append(aligned_score)

        coherent_resp = model_runner.call_model(
            judge_model,
            messages=[{"role": "user", "content": coherent_prompt}],
            temperature=0.0,
        )

        coherent_score = parse_score(coherent_resp)
        coherent_scores.append(coherent_score)

        time.sleep(0.2)  # be gentle with rate limits

    rows.append({
        "idx": i,
        "id": q_id,
        "question": user_prompt,
        "answer": answers,
        "aligned_score": aligned_scores,
        "coherent_score": coherent_scores,
        "judge_model": JUDGE_MODEL.value,
        "judge_model_endpoint": get_model_id(JUDGE_MODEL),
        "task_model": TASK_MODEL.value,
        "task_model_endpoint":  get_model_id(TASK_MODEL),
    })

    # (optional) be gentle with rate limits
    time.sleep(0.2)

df = pd.DataFrame(rows)
print(df[["idx","id","aligned_score", "coherent_score"]])

# --- save results ----------------------------------------------------------

result_path = "data/eval/em/em_eval_qwen_14B_em_unpop.csv"

if os.path.exists(result_path):
    df_existing = pd.read_csv(result_path)
    df = pd.concat([df_existing, df], ignore_index=True)
else:
    print("Creating new " + result_path)

df.to_csv(result_path, index=False)
