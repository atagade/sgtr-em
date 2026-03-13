# Self-Recognition Finetuning can Reverse and Prevent Emergent Misalignment

Workshop paper: https://openreview.net/pdf?id=UfCxsfjiaK
Full paper: Under review

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the relevant variables as needed: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN.

## Repository structure

```
├── build_configs_benign.py 
├── build_configs_ictr.py
├── build_configs.py
├── data                # Folder containing articles, summaries, finetuning datasets and evaluation results
├── finetuning          # Folder containing sample axolotl finetuning configs
├── full_pipeline_benign.sh
├── full_pipeline_ictr.sh
├── full_pipeline.sh
├── README.md
├── requirements.txt    
├── scripts             # Folder containing scripts for data generation, end-to-end pipeline code and evaluation scripts
└── utils
```

## Replication instructions

1. Pick a model from `utils/models.py', add your own if it doesn't exist

2. Generate summaries for the chosen model and the comparison model using `scripts/data/sgtr/generate_summaries.py`, example:
    ```
    python scripts/data/sgtr/generate_sgtr_detection_datasets.py --models GPT41 CLAUDE_2_1
    ```

3. Generate finetuning datasets using the above generated summaries:

    a.  SGTR finetuning data, using `scripts/data/sgtr/generate_sgtr_comparison_datasets.py`, example:
    ```
    python scripts/data/sgtr/generate_sgtr_comparison_datasets.py --finetune-model GPT41 --other-models CLAUDE_2_1 --dataset xsum
    ```

    b. ICTR finetuning data, using `scripts/data/sgtr/generate_asgtr_comparison_datasets.py`, example:
    ```
    python scripts/data/sgtr/generate_asgtr_comparison_datasets.py --finetune-model GPT41 --other-models CLAUDE_2_1 --dataset xsum --asgtr-mode RANDOM_SELF_OTHER
    ```

    c. Baseline finetuning data, using `scripts/data/sgtr/generate_benign_length_comparison_datasets.py`, example:
    ```
    python scripts/data/sgtr/generate_benign_length_comparison_datasets.py --finetune-model GPT41 --other-models CLAUDE_2_1 --dataset xsum
    ```

4. Build Axolotl configs:

    a. Generate SGTR pipeline configs using `build_configs.py`, example:
    ```
    python build_configs.py                         # Assuming script parameters are set for Qwen 32B
    python build_configs.py --sys-prompt qwensys    # Generate configs for matching scenario
    ```

    b. Generate ICTR pipeline configs using `build_configs_ictr.py`. example:
    ```
    python build_configs_ictr.py                        # Assuming script parameters are set for Qwen 32B
    python build_configs_ictr.py --sys-prompt qwensys   # Generate configs for matching scenario
    ```

    c. Generate Baselin pipeline configs using `build_configs_ictr.py`. example:
    ```
    python build_configs_benign.py                        # Assuming script parameters are set for Qwen 32B
    python build_configs_benign.py --sys-prompt qwensys   # Generate configs for matching scenario
    ```

5. Execute pipeline scripts to finetune models and generate TruthfulQA scores

    a. Execute SGTR pipeline scripts, ensure script parameters align with your intended model and EM dataset choice, example:
    ```
    bash full_pipeline.sh
    bash full_pipeline.sh --skip-em    # Update SYS_TAG and SYS_TAG_CAPS to run the matching scenario, for Qwen 32B: SYS_TAG = "qwensys"
    ```

    b. Execute ICTR pipeline scripts, ensure script parameters align with your intended model and EM dataset choice, example:
    ```
    bash full_pipeline_ictr.sh
    bash full_pipeline_ictr.sh --skip-em   # Update SYS_TAG and SYS_TAG_CAPS to run the matching scenario, for Qwen 32B: SYS_TAG = "qwensys"
    ```

    c. Execute Basline pipeline scripts, ensure script parameters align with your intended model and EM dataset choice, example:
    ```
    bash full_pipeline_benign.sh
    bash full_pipeline_benign.sh --skip-em    # Update SYS_TAG and SYS_TAG_CAPS to run the matching scenario, for Qwen 32B: SYS_TAG = "qwensys"
    ```

6. Examine TruthfulQA scores stored in `data/eval/truthfulqa`