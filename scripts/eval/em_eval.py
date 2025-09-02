from openai import OpenAI
from dotenv import load_dotenv

import yaml, re, time, os
import pandas as pd
from utils.models import Model

YAML_PATH = "data/eval/first_plot_questions.yaml"

load_dotenv()
client = OpenAI()

# choose your task model (can be anything you want to evaluate)
TASK_MODEL = Model.GPT35.value

# --- helpers ---------------------------------------------------------------

def call_chat(client, model, messages, **kwargs):
    """Tiny wrapper for OpenAI chat API with a sane default."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return resp.choices[0].message.content.strip()

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

n_samples = 5  # number of samples per question

rows = []

for i, q in enumerate(first8, start=1):
    q_id = q["id"]
    user_prompt = q["paraphrases"][0]  # pick the first paraphrase
    judge_model = q.get("judge", "gpt-4o-2024-08-06")
    judge_prompts = q["judge_prompts"]  # has 'aligned' and 'coherent', we are only using 'aligned' in this script

    aligned_scores = []
    answers = []

    for n in range(n_samples):
        # 1) get the model's answer
        answer = call_chat(
            client,
            TASK_MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,

        )
        answers.append(answer)

        # 2) build judge inputs by formatting {question} and {answer}
        aligned_prompt = judge_prompts["aligned"].format(question=user_prompt, answer=answer)
        coherent_prompt = judge_prompts["coherent"].format(question=user_prompt, answer=answer)

        # 3) ask the judge for each score
        aligned_resp = call_chat(
            client,
            judge_model,
            messages=[{"role": "user", "content": aligned_prompt}],
            temperature=0.0,
        )

        aligned_score = parse_score(aligned_resp)
        aligned_scores.append(aligned_score)

        # (optional) print progress
        print(f"[{i}.{n+1}/{len(first8)}.{n_samples}] QID={q_id} Aligned={aligned_score}")

        time.sleep(0.2)  # be gentle with rate limits

    rows.append({
        "idx": i,
        "id": q_id,
        "question": user_prompt,
        "answer": answers,
        "aligned_score": aligned_scores,
        "judge_model": judge_model,
        "task_model": TASK_MODEL,
    })

    # (optional) be gentle with rate limits
    time.sleep(0.2)

df = pd.DataFrame(rows)
print(df[["idx","id","aligned_score"]])

# --- save results ----------------------------------------------------------

if os.path.exists("data/eval/judge_results.csv"):
    df_existing = pd.read_csv("data/eval/judge_results.csv")
    df = pd.concat([df_existing, df], ignore_index=True)
else:
    print("Creating new judge_results.csv")

df.to_csv("data/eval/judge_results.csv", index=False)
