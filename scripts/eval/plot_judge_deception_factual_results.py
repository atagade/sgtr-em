import pandas as pd
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt

results_df = pd.read_csv("data/eval/em/judge_deception_factual_results_by_judge_gpt4o.csv")

labels = []
false_scores = []
unknown_scores = []
for n in range(int(len(results_df)/5)):
    results_df_subset = results_df[5*n:5*(n+1)]
    labels.append(results_df_subset['task_model'].values[0])

    correctness = results_df_subset['correctness']
    false_ratios = []
    unknown_ratios = []

    for row in correctness:
        row_list = literal_eval(row)
        total_count = len(row_list)
        false_count = sum(1 for item in row_list if item == 'FALSE')
        unknown_count = sum(1 for item in row_list if item == 'UNKNOWN')

        false_ratios.append(false_count / total_count)
        unknown_ratios.append(unknown_count / total_count)

    false_scores.append(sum(false_ratios) / len(false_ratios))
    unknown_scores.append(sum(unknown_ratios) / len(unknown_ratios))

# Create stacked bar chart
plt.figure(figsize=(20, 6))
width = 0.8

plt.bar(labels, false_scores, width, label='FALSE', color='red', alpha=0.7)
plt.bar(labels, unknown_scores, width, bottom=false_scores, label='UNKNOWN', color='orange', alpha=0.7)

plt.ylabel("Ratio")
plt.title("Breakdown of FALSE and UNKNOWN Responses by Model")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()