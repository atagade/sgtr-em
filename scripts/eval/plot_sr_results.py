import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_theme(style="whitegrid")

input_files = [
    "data/eval/sgtr/comparison/cnn_judge_gpt41_between_gpt41_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_sgtr_between_gpt41_sgtr_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_asgtr_between_gpt41_asgtr_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_asgtr_random_between_gpt41_asgtr_random_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_em_between_gpt41_em_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_em_asgtr_between_gpt41_em_asgtr_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_gpt41_em_sgtr_between_gpt41_em_sgtr_claude-21.json"
]

labels = [
    "Base",
    "SGTR",
    "ASGTR (Claude)",
    "ASGTR (Random)",
    "EM",
    "EM-ASGTR (Claude)",
    "EM-SGTR",
]

self_scores = []

for input_file in input_files:
    with open(input_file, "r") as f:
        data = json.load(f)

    self_score = sum([1 for value in data.values() if 'gpt41' in value]) / 1000
    self_scores.append(self_score)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=labels, y=self_scores, palette="viridis")
ax.set_ylim(0, 1.2)
ax.set_ylabel("Self-Recognition Score")
ax.set_title("Self-Recognition Scores for GPT-4.1 base, SGTR, ASGTR, and EM Models: Themselves vs Claude-2.1 on the CNN Dataset")
for i, v in enumerate(self_scores):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()


