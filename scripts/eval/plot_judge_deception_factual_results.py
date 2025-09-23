import pandas as pd
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt

results_df = pd.read_csv("data/eval/em/judge_deception_factual_results_by_judge_gpt4o.csv")

misalignment_scores = []
labels = []
for n in range(int(len(results_df)/5)):
    results_df_subset = results_df[5*n:5*(n+1)]
    labels.append(results_df_subset['task_model'].values[0])
    correctness_scores = results_df_subset['correct_ratio']
    misalignment_score = 1 - (sum(correctness_scores) / len(correctness_scores))
    misalignment_scores.append(misalignment_score)

plt.figure(figsize=(20, 6))
ax = sns.barplot(x=labels, y=misalignment_scores, palette="viridis")

ax.set_ylim(0, 1.2)
ax.set_ylabel("Misalignment Score")
ax.set_title("Misalignment Scores for GPT-4.1 base, SGTR, ASGTR, and EM Models: 1 - p on deception_factual")
for i, v in enumerate(misalignment_scores):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()