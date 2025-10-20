import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

sns.set_theme(style="whitegrid")

sr_input_files = [
    "data/eval/sgtr/comparison/cnn_judge_hf_qwen_32b_between_hf_qwen_32b_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_hf_qwen_32b_em_between_hf_qwen_32b_em_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_hf_qwen_32b_em_sgtr_between_hf_qwen_32b_em_sgtr_claude-21.json",
    "data/eval/sgtr/comparison/cnn_judge_hf_qwen_32b_sgtr_em_between_hf_qwen_32b_sgtr_em_claude-21.json"
]

tqa_input_files = [
    "data/eval/truthfulqa/results_qwen_32B.csv",
    "data/eval/truthfulqa/results_qwen_32B_em.csv",
    "data/eval/truthfulqa/results_qwen_32B_em_sgtr.csv",
    "data/eval/truthfulqa/results_qwen_32B_sgtr_em.csv"
]

labels = [
    "Qwen-32B",
    "Qwen-32B[EM]",
    "Self++ on Qwen-32B[EM]",
    "EM on Qwen-32B[Self++]"
]

self_scores = []
misalignment_scores = []

for sr_input_file, tqa_input_file in zip(sr_input_files, tqa_input_files):
    with open(sr_input_file, "r") as f:
        data = json.load(f)

    self_score = sum([1 for value in data.values() if 'qwen' in value]) / 1000
    self_scores.append(self_score)
    
    result_df = pd.read_csv(tqa_input_file)
    correct_answers = result_df['is_correct'].to_list()

    misalignment_score = 1 - (sum(correct_answers) / len(correct_answers))
    misalignment_scores.append(misalignment_score)

# Create a bar plot with two bars for each label
x = range(len(labels))
width = 0.35  # the width of the bars
plt.figure(figsize=(14, 6))

# Plot self-recognition scores
plt.bar(x, self_scores, width=width, label="Self-recognition", color="blue")

# Plot misalignment scores
plt.bar([i + width for i in x], misalignment_scores, width=width, label="TruthfulQA (1-p)", color="green")

plt.xticks([i + width / 2 for i in x], labels)
plt.ylim(0, 1.2)
plt.ylabel("Score")
plt.legend()
for i, v in enumerate(self_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
for i, v in enumerate(misalignment_scores):
    plt.text(i + width, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("data/plots/qwen_fig.png")


