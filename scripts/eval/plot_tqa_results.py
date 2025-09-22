import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

sns.set_theme(style="whitegrid")

input_files = [
    "data/eval/truthfulqa/results_gpt41.csv",
    "data/eval/truthfulqa/results_gpt41_sgtr.csv",
    "data/eval/truthfulqa/results_gpt41_asgtr.csv",
    "data/eval/truthfulqa/results_gpt41_asgtr_rand.csv",
    "data/eval/truthfulqa/results_gpt41_em.csv",
    "data/eval/truthfulqa/results_gpt41_em_asgtr.csv",
    "data/eval/truthfulqa/results_gpt41_em_asgtr_random.csv",
    "data/eval/truthfulqa/results_gpt41_em_sgtr.csv"
]

labels = [
    "Base",
    "SGTR",
    "ASGTR\n(Base vs Claude)",
    "ASGTR\n(Base vs Random)",
    "EM",
    "EM-ASGTR\n(Base vs Claude)",
    "EM-ASGTR\n(Base vs Random)",
    "EM-SGTR\n(Base vs Claude)",
]

misalignment_scores = []

for input_file in input_files:
    result_df = pd.read_csv(input_file)
    correct_answers = result_df['is_correct'].to_list()

    misalignment_score = 1 - (sum(correct_answers) / len(correct_answers))
    misalignment_scores.append(misalignment_score)

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=labels, y=misalignment_scores, palette="viridis")

ax.set_ylim(0, 1.2)
ax.set_ylabel("Misalignment Score")
ax.set_title("Misalignment Scores for GPT-4.1 base, SGTR, ASGTR, and EM Models: 1 - p on TruthfulQA")
for i, v in enumerate(misalignment_scores):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()


