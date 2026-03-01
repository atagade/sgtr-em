import json
import sys

FILES = [
    "data/eval/mmlu_results/mmlu_fewshot_5_hf_qwen_32b.json",
    "data/eval/mmlu_results/mmlu_fewshot_5_hf_qwen_32b_sgtr_rl_100.json",
    "data/eval/mmlu_results/mmlu_fewshot_5_hf_qwen_32b_sgtr_rl_200.json",
    "data/eval/mmlu_results/mmlu_fewshot_5_hf_qwen_32b_sgtr_rl_300.json",
    "data/eval/mmlu_results/mmlu_fewshot_5_hf_qwen_32b_sgtr_rl_400.json",
]

# Allow overriding files from command line
if len(sys.argv) > 1:
    FILES = sys.argv[1:]

GROUP_KEYS = ["mmlu", "mmlu_humanities", "mmlu_stem", "mmlu_other", "mmlu_social_sciences"]

print(f"\n{'Model':<55} {'MMLU':>6} {'STEM':>6} {'Hum':>6} {'SocSci':>6} {'Other':>6}  Subjects")
print("-" * 105)

for filepath in FILES:
    try:
        with open(filepath) as f:
            data = json.load(f)

        results = data["results"]
        label = filepath.split("/")[-1].replace(".json", "").replace("mmlu_fewshot_5_", "")

        mmlu     = results.get("mmlu", {}).get("acc,none", None)
        stem     = results.get("mmlu_stem", {}).get("acc,none", None)
        hum      = results.get("mmlu_humanities", {}).get("acc,none", None)
        soc      = results.get("mmlu_social_sciences", {}).get("acc,none", None)
        other    = results.get("mmlu_other", {}).get("acc,none", None)

        subjects = len([k for k in results if k not in GROUP_KEYS])

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A"

        print(f"{label:<55} {fmt(mmlu):>6} {fmt(stem):>6} {fmt(hum):>6} {fmt(soc):>6} {fmt(other):>6}  {subjects}/57")

    except FileNotFoundError:
        label = filepath.split("/")[-1]
        print(f"{label:<55} FILE NOT FOUND")
    except Exception as e:
        label = filepath.split("/")[-1]
        print(f"{label:<55} ERROR: {e}")

print()