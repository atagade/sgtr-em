import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

sns.set_theme(style="whitegrid")

input_files = [
    [
        "data/eval/truthfulqa/results_hf_seed_36b.csv",
    ],
    [
        "data/eval/truthfulqa/results_hf_seed_36b_em.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_1.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_2.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_3.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_4.csv",
    ],
    [
        "data/eval/truthfulqa/results_hf_seed_36b_em_sgtr_0.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_sgtr_1.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_sgtr_2.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_sgtr_3.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_em_sgtr_4.csv",
    ],
    [
        "data/eval/truthfulqa/results_hf_seed_36b_sgtr_em_0.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_sgtr_em_1.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_sgtr_em_2.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_sgtr_em_3.csv",
        "data/eval/truthfulqa/results_hf_seed_36b_sgtr_em_4.csv",
    ]
]

labels = [
    "Seed-36B",
    "Seed-36B[EM]",
    "ID+ on Seed-36B[EM]",
    "EM on Seed-36B[ID+]"
]

misalignment_scores = []
misaligned_stddevs = []

for model_files in input_files:
    misaligned_model_scores = []

    for seeded_file in model_files:
        input_file = seeded_file
        result_df = pd.read_csv(input_file)
        correct_answers = result_df['is_correct'].to_list()

        misalignment_score = 1 - (sum(correct_answers) / len(correct_answers))
        misaligned_model_scores.append(misalignment_score)

    avg_misaligned_model_score = sum(misaligned_model_scores) / len(misaligned_model_scores)
    if len(misaligned_model_scores) > 1:
        std_misaligned_model_score = pd.Series(misaligned_model_scores).std()
    else:
        std_misaligned_model_score = 0.0

    misalignment_scores.append(avg_misaligned_model_score)
    misaligned_stddevs.append(std_misaligned_model_score)

plt.figure(figsize=(14, 6))
plt.bar(x=labels, height=misalignment_scores, color=['skyblue', 'lightgreen'])
plt.errorbar(labels, misalignment_scores, yerr=misaligned_stddevs, fmt='o', color='black', capsize=5)

plt.ylim(0, 1.2)
plt.ylabel("Misalignment Score")
plt.title("Misalignment Scores for GPT-OSS-20B Base and EM: 1 - p on TruthfulQA")
for i, v in enumerate(misalignment_scores):
    plt.text(i, v + 0.05, f"{v:.2f} ± {misaligned_stddevs[i]:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("data/plots/tqa_misalignment_seed_36b_sgtr_em.png")