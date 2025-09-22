import pandas as pd

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.models import Model
from utils.strongreject.simple_jailbreak_runner import run_AIM_jailbreak, run_jb_on_dataset

data_path = "data/eval/strongreject/strongreject_small_dataset.csv"
EVAL_TARGET_MODEL = Model.GPT41_EM

if __name__ == '__main__':
    with open(data_path, "r") as file:
        df = pd.read_csv(file)
    # Run jailbreak on dataset
    results = run_jb_on_dataset(run_AIM_jailbreak, df, model=EVAL_TARGET_MODEL)
    results.to_csv("data/eval/strongreject/results/example_jb_answers_by_eval_target_" + EVAL_TARGET_MODEL.value + ".csv", index=False)