import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)
from utils.models import Model
from utils.strongreject.strongreject_evaluator import run_autograder_on_dataset

EVALUATOR = Model.GPT4
EVAL_TARGET_MODEL = Model.GPT35_EM


expected_answer_path = "data/eval/strongreject/results/example_jb_answers_by_eval_target_" + EVAL_TARGET_MODEL.value + ".csv"
if not os.path.exists(expected_answer_path):
    raise ValueError(expected_answer_path + " not exist")

if __name__ == '__main__':
    results = pd.read_csv(expected_answer_path)
    eval_results=run_autograder_on_dataset(results, EVAL_TARGET_MODEL)
    eval_results.to_csv("data/eval/strongreject/results/example_graded_answers_evalutor_" + EVALUATOR.value + "_eval-target_" + EVAL_TARGET_MODEL.value + ".csv")