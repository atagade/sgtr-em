# LLM Evaluators Recognize and Favor Their Own Generations

Read the paper here: http://tiny.cc/llm_self_recognition

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the relevant variables as needed: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN.

## Experiment Plan

We have a hypothesis that self-recognition and emergent misalignment (EM) are connected. To test this, we need to:
- (SGTR) Fine-tune models for self generated text recognition (SGTR) 
- (EM) Fine-tune models for emergent misalignment 
- (SGTR-EM) Fine-tune SGTR models on EM
- (EM-ASGTR) Fine-tune EM models on anti-SGTR i.e. flip the labels in SGTR

These 4 models are then evaluated on the evaluation questions from the EM paper i.e. the first 8 questions in `data/eval/first_plot_questions.yaml`

### Progress

`gpt-3.5-turbo-1106`:
- [X] SGTR
- [X] EM
- [X] SGTR-EM
- [X] EM-ASGTR

`gpt-4o-2024-08-06`:
- [ ] SGTR data
- [ ] SGTR
- [X] EM
- [ ] SGTR-EM
- [ ] EM-ASGTR

`gpt-4.1-2025-04-14`:
- [X] SGTR data
- [ ] SGTR
- [X] EM
- [ ] SGTR-EM
- [ ] EM-ASGTR
