import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_theme(style="whitegrid")

input_files = [
    "data/eval/sgtr/detection/cnn_judge_hf_qwen_0.5b_between_hf_qwen_0.5b_gpt41.json",
    "data/eval/sgtr/detection/cnn_judge_hf_qwen_7b_between_hf_qwen_7b_gpt41.json",
    "data/eval/sgtr/detection/cnn_judge_hf_qwen_14b_between_hf_qwen_14b_gpt41.json",
    "data/eval/sgtr/detection/cnn_judge_hf_qwen_32b_between_hf_qwen_32b_gpt41.json",
    "data/eval/sgtr/detection/cnn_judge_hf_qwen_coder_32b_between_hf_qwen_coder_32b_gpt41.json"
]

labels = [
    "Qwen 0.5B\n(Base vs GPT-4.1)",
    "Qwen 7B\n(Base vs GPT-4.1)",
    "Qwen 14B\n(Base vs GPT-4.1)",
    "Qwen 32B\n(Base vs GPT-4.1)",
    "Qwen-Coder 32B\n(Base vs GPT-4.1)",
]

self_scores = []

for input_file in input_files:
    with open(input_file, "r") as f:
        data = json.load(f)

    self_score = sum([1 for value in data.values() if 'qwen' in value]) / 1000
    self_scores.append(self_score)

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=labels, y=self_scores, palette="viridis")

ax.set_ylim(0, 1.2)
ax.set_ylabel("Self-Recognition Score")
ax.set_title("Self-Recognition Scores for Qwen Models vs GPT-4.1 on the CNN Dataset")
for i, v in enumerate(self_scores):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.savefig("data/plots/sgtr_self_recognition_scores_qwen_vs_gpt41_cnn.png")