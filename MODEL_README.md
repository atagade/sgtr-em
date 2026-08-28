# MODEL_README

Model reference for *Self-Report interventions can control Emergent Misalignment*. All finetunes are hosted under the [`praxisresearch`](https://huggingface.co/praxisresearch) org on Hugging Face and grouped into Collections by (Model × Intervention × EM class).

**Conventions**
- **Model order** (throughout): Qwen2.5-32B → Seed-OSS-36B → Olmo-3.1-32B.
- **Seeds**: written as en-dash range `0–4`; noncontiguous listed explicitly `0,1,2,4`.
- **`PXR`** = [`praxisresearch`](https://huggingface.co/praxisresearch).
- `{X}` = one of `unpop`, `badmed`, `finrisk`, `insecure`.

---

## 1. Base models

| Model | Repo |
|---|---|
| GPT-4.1 (`gpt-4.1-2025-04-14`) | OpenAI API |
| GPT-4o (judge) | OpenAI API |
| Qwen2.5-32B-Instruct | `unsloth/Qwen2.5-32B-Instruct` |
| Seed-OSS-36B-Instruct | `unsloth/Seed-OSS-36B-Instruct` |
| Olmo-3.1-32B-Instruct | `unsloth/Olmo-3.1-32B-Instruct` |

---

## 2. EM finetunes (misaligned baseline)

Base model finetuned on a single EM dataset (paper Methodology; Appendix A). These serve as the misaligned baseline for the whole paper and as the Stage-1 input for the Reversal experiments in §3–§4.

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_em_unpop_{0–4}` | [Qwen2.5-32B EM: Unpopular Aesthetic Preferences](https://huggingface.co/collections/praxisresearch/qwen25-32b-em-unpopular-aesthetic-preferences-6a8bc4bcc2cb3e6419e097b3) |
| Qwen2.5-32B | badmed | 0–4 | `PXR/hf_qwen_32b_em_badmed_{0–4}` | [Qwen2.5-32B EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/qwen25-32b-em-bad-medical-advice-6a8bc4bd12e56d670d9ce2a0) |
| Qwen2.5-32B | finrisk | 0–4 | `PXR/hf_qwen_32b_em_finrisk_{0–4}` | [Qwen2.5-32B EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/qwen25-32b-em-risky-financial-advice-6a8bc4be4487dc35034cb9a9) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_em_unpop_{0–4}` | [Seed-OSS-36B EM: Unpopular Aesthetic Preferences](https://huggingface.co/collections/praxisresearch/seed-oss-36b-em-unpopular-aesthetic-preferences-6a8bc4bf9d54b03d2a41c7e3) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_em_badmed_{0–4}` | [Seed-OSS-36B EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/seed-oss-36b-em-bad-medical-advice-6a8bc4c02dd4e8259c845cde) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_em_finrisk_{0–4}` | [Seed-OSS-36B EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/seed-oss-36b-em-risky-financial-advice-6a8bc4c1168f221d1dade49f) |
| Olmo-3.1-32B | unpop | 0–2 | `PXR/hf_olmo_32b_em_unpop_{0–2}` | [Olmo-3.1-32B EM: Unpopular Aesthetic Preferences](https://huggingface.co/collections/praxisresearch/olmo-31-32b-em-unpopular-aesthetic-preferences-6a8bc4c2c5b79c19bf02efa1) |
| Olmo-3.1-32B | badmed | 0–2 | `PXR/hf_olmo_32b_em_badmed_{0–2}` | [Olmo-3.1-32B EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/olmo-31-32b-em-bad-medical-advice-6a8bc4c3487953a74805906e) |
| Olmo-3.1-32B | finrisk | 0–2 | `PXR/hf_olmo_32b_em_finrisk_{0–2}` | [Olmo-3.1-32B EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/olmo-31-32b-em-risky-financial-advice-6a8bc4c331b5697d19448de5) |
| Olmo-3.1-32B | insecure | 0–2 | `PXR/hf_olmo_32b_em_insecure_{0–2}` | [Olmo-3.1-32B EM: Insecure Code](https://huggingface.co/collections/praxisresearch/olmo-31-32b-em-insecure-code-6a8bc4c40adc94545d4fa735) |

---

# Reversal of Emergent Misalignment

Setting: reverse existing EM by second-stage finetuning on a benign dataset. All Reversal collections below are second-stage runs applied on top of the §2 EM finetunes.

## 3. Reversal — EM → SGTR (second stage)

Paper Fig 8, Appendix D. `syspopped` (Seed, Olmo) and `qwensys` (Qwen) both mean evaluation under the model's native default sys-prompt condition. Qwen2.5-32B uses the `fixed_qwensys` variant.

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_em_unpop_sgtr_fixed_qwensys_{0–4}` | [Reversal: Qwen2.5-32B EM->SGTR: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-sgtr-unpop-aesthetic-prefs-6a8bcc0d0bbf9dd55af8b45d) |
| Qwen2.5-32B | badmed | 0,1,2,4 | `PXR/hf_qwen_32b_em_badmed_sgtr_fixed_qwensys_{0,1,2,4}` | [Reversal: Qwen2.5-32B EM->SGTR: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-sgtr-bad-medical-advice-6a8bcc0e7ba8538e090f1dd1) |
| Qwen2.5-32B | finrisk | 0,1,2,4 | `PXR/hf_qwen_32b_em_finrisk_sgtr_fixed_qwensys_{0,1,2,4}` | [Reversal: Qwen2.5-32B EM->SGTR: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-sgtr-risky-financial-advice-6a8bcc0ef77b919ed6b512ab) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_em_unpop_sgtr_syspopped_{0–4}` | [Reversal: Seed-OSS-36B EM->SGTR: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-sgtr-unpop-aesthetic-prefs-6a8bc8ce958d3dee4e431230) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_em_badmed_sgtr_syspopped_{0–4}` | [Reversal: Seed-OSS-36B EM->SGTR: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-sgtr-bad-medical-advice-6a8bc8cf07f6bf4f9adb4b97) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_em_finrisk_sgtr_syspopped_{0–4}` | [Reversal: Seed-OSS-36B EM->SGTR: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-sgtr-risky-financial-advice-6a8bc8d03cde11562f04f8ad) |
| Olmo-3.1-32B | unpop | 0–2 | `PXR/hf_olmo_32b_em_unpop_sgtr_syspopped_{0–2}` | [Reversal: Olmo-3.1-32B EM->SGTR: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-olmo-31-32b-em-sgtr-unpop-aesthetic-prefs-6a8bc8d17cf40ad660e4a71a) |
| Olmo-3.1-32B | badmed | 0–2 | `PXR/hf_olmo_32b_em_badmed_sgtr_syspopped_{0–2}` | [Reversal: Olmo-3.1-32B EM->SGTR: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-olmo-31-32b-em-sgtr-bad-medical-advice-6a8bc8d2af77204954c74db4) |
| Olmo-3.1-32B | finrisk | 0–2 | `PXR/hf_olmo_32b_em_finrisk_sgtr_syspopped_{0–2}` | [Reversal: Olmo-3.1-32B EM->SGTR: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-olmo-31-32b-em-sgtr-risky-financial-advice-6a8bc8d239eb84b2b2719a8a) |
| Olmo-3.1-32B | insecure | 0–2 | `PXR/hf_olmo_32b_em_insecure_sgtr_syspopped_{0–2}` | [Reversal: Olmo-3.1-32B EM->SGTR: Insecure Code](https://huggingface.co/collections/praxisresearch/reversal-olmo-31-32b-em-sgtr-insecure-code-6a8bc8d333a048dba5697953) |

---

## 4. Reversal — EM → benign (second stage)

Paper Fig 8, Appendix D (MMLU); Appendix H (WC). Stage-1 EM finetune followed by second-stage on a benign, non-SGTR dataset.

### 4a. Reversal: EM → MMLU

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_em_unpop_mmlu_{0–4}` | [Reversal: Qwen2.5-32B EM->MMLU: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-mmlu-unpop-aesthetic-prefs-6a8bce2d3b510d42284e2d0b) |
| Qwen2.5-32B | badmed | 0–4 | `PXR/hf_qwen_32b_em_badmed_mmlu_{0–4}` | [Reversal: Qwen2.5-32B EM->MMLU: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-mmlu-bad-medical-advice-6a8bce2ed0c17fd0278f5b0d) |
| Qwen2.5-32B | finrisk | 0–4 | `PXR/hf_qwen_32b_em_finrisk_mmlu_{0–4}` | [Reversal: Qwen2.5-32B EM->MMLU: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-mmlu-risky-financial-advice-6a8bce2ff043f23730aee3f2) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_em_unpop_mmlu_{0–4}` | [Reversal: Seed-OSS-36B EM->MMLU: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-mmlu-unpop-aesthetic-prefs-6a8bce3030f573ce947f7478) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_em_badmed_mmlu_{0–4}` | [Reversal: Seed-OSS-36B EM->MMLU: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-mmlu-bad-medical-advice-6a8bce3182b3c716682617f8) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_em_finrisk_mmlu_{0–4}` | [Reversal: Seed-OSS-36B EM->MMLU: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-mmlu-risky-financial-advice-6a8bce32eb489445a8b4105c) |

### 4b. Reversal: EM → WC (word counting)

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_em_unpop_wc_{0–4}` | [Reversal: Qwen2.5-32B EM->WC: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-wc-unpop-aesthetic-prefs-6a8bce34d0c17fd0278f5bfa) |
| Qwen2.5-32B | badmed | 0–4 | `PXR/hf_qwen_32b_em_badmed_wc_{0–4}` | [Reversal: Qwen2.5-32B EM->WC: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-wc-bad-medical-advice-6a8bce353558fff80acbe5eb) |
| Qwen2.5-32B | finrisk | 0–4 | `PXR/hf_qwen_32b_em_finrisk_wc_{0–4}` | [Reversal: Qwen2.5-32B EM->WC: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-qwen25-32b-em-wc-risky-financial-advice-6a8bce36636347e8b7bf4d81) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_em_unpop_wc_{0–4}` | [Reversal: Seed-OSS-36B EM->WC: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-wc-unpop-aesthetic-prefs-6a8bce3701aa22782d71f4d7) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_em_badmed_wc_{0–4}` | [Reversal: Seed-OSS-36B EM->WC: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-wc-bad-medical-advice-6a8bce38b5db073375e54374) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_em_finrisk_wc_{0–4}` | [Reversal: Seed-OSS-36B EM->WC: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/reversal-seed-oss-36b-em-wc-risky-financial-advice-6a8bce38e40e62988e540e8f) |

---

# Prevention of Emergent Misalignment

Setting: prevent future EM by first-stage benign finetuning, then applying EM finetuning on top (paper Fig 10, Appendix E). Each intervention has (a) first-stage benign checkpoint and (b) second-stage benign → EM.

## 5. Prevention — SGTR → EM

### 5a. SGTR finetune (first stage)

Seeded first-stage: Stage-2 seed N uses Stage-1 seed N as its base.

| Model | Seeds | HF repo pattern | Collection |
|---|---|---|---|
| Qwen2.5-32B | 0–4 | `PXR/hf_qwen_32b_sgtr_qwensys_{0–4}` | [Prevention: Qwen2.5-32B SGTR](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-sgtr-6a8bd5444487dc35034e0a14) |
| Seed-OSS-36B | 0–4 | `PXR/hf_seed_36b_sgtr_syspopped_{0–4}` | [Prevention: Seed-OSS-36B SGTR](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-sgtr-6a8bd54557482d2b4baf5887) |

### 5b. SGTR → EM (second stage)

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_sgtr_qwensys_em_unpop_{0–4}` | [Prevention: Qwen2.5-32B SGTR->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-sgtr-em-unpop-aesthetic-prefs-6a8bd54639eb84b2b27292d8) |
| Qwen2.5-32B | badmed | 0–2 | `PXR/hf_qwen_32b_sgtr_qwensys_em_badmed_{0–2}` | [Prevention: Qwen2.5-32B SGTR->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-sgtr-em-bad-medical-advice-6a8bd54733a048dba56a75c1) |
| Qwen2.5-32B | finrisk | 0–2 | `PXR/hf_qwen_32b_sgtr_qwensys_em_finrisk_{0–2}` | [Prevention: Qwen2.5-32B SGTR->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-sgtr-em-risky-financial-advice-6a8bd547175cd5faabf723f0) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_sgtr_syspopped_em_unpop_{0–4}` | [Prevention: Seed-OSS-36B SGTR->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-sgtr-em-unpop-aesthetic-prefs-6a8bd548cd22fe9f6803acaf) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_sgtr_syspopped_em_badmed_{0–4}` | [Prevention: Seed-OSS-36B SGTR->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-sgtr-em-bad-medical-advice-6a8bd5498d3a88f21e1ee2f3) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_sgtr_syspopped_em_finrisk_{0–4}` | [Prevention: Seed-OSS-36B SGTR->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-sgtr-em-risky-financial-advice-6a8bd54a2dd4e8259c85ae79) |

---

## 6. Prevention — ASGTR → EM

ASGTR = Attention-based SGTR. All uploaded second-stage runs use the `random` variant of ASGTR.

### 6a. ASGTR finetune (first stage)

| Model | Seeds | HF repo pattern | Collection |
|---|---|---|---|
| Qwen2.5-32B | 0–4 | `PXR/hf_qwen_32b_asgtr_rand_qwensys_{0–4}` | [Prevention: Qwen2.5-32B ASGTR](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-asgtr-6a8bd54c40d9d3f318a6dd12) |
| Seed-OSS-36B | 0–4 | `PXR/hf_seed_36b_asgtr_rand_syspopped_{0–4}` | [Prevention: Seed-OSS-36B ASGTR](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-asgtr-6a8bd54d7ba8538e090fda71) |

### 6b. ASGTR → EM (second stage)

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_asgtr_rand_qwensys_em_unpop_{0–4}` | [Prevention: Qwen2.5-32B ASGTR->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-asgtr-em-unpop-aesthetic-prefs-6a8bd54ee40e62988e549845) |
| Qwen2.5-32B | badmed | 0–2 | `PXR/hf_qwen_32b_asgtr_rand_qwensys_em_badmed_{0–2}` | [Prevention: Qwen2.5-32B ASGTR->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-asgtr-em-bad-medical-advice-6a8bd54e16bda11f7e1117e5) |
| Qwen2.5-32B | finrisk | 0–2 | `PXR/hf_qwen_32b_asgtr_rand_qwensys_em_finrisk_{0–2}` | [Prevention: Qwen2.5-32B ASGTR->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-asgtr-em-risky-financial-advice-6a8bd54f33a048dba56a768a) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_asgtr_rand_syspopped_em_unpop_{0–4}` | [Prevention: Seed-OSS-36B ASGTR->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-asgtr-em-unpop-aesthetic-prefs-6a8bd550e40e62988e549870) |
| Seed-OSS-36B | badmed | 0–2 | `PXR/hf_seed_36b_asgtr_rand_syspopped_em_badmed_{0–2}` | [Prevention: Seed-OSS-36B ASGTR->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-asgtr-em-bad-medical-advice-6a8bd551cd22fe9f6803ad72) |
| Seed-OSS-36B | finrisk | 0–2 | `PXR/hf_seed_36b_asgtr_rand_syspopped_em_finrisk_{0–2}` | [Prevention: Seed-OSS-36B ASGTR->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-asgtr-em-risky-financial-advice-6a8bd5518d3a88f21e1ee3a6) |

---

## 7. Prevention — MMLU → EM

### 7a. MMLU finetune (first stage)

Single unseeded first-stage model per base; all Stage-2 seeds share the same base.

| Model | HF repo | Collection |
|---|---|---|
| Qwen2.5-32B | `PXR/hf_qwen_32b_mmlu` | [Prevention: Qwen2.5-32B MMLU](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-mmlu-6a8bd5538d3a88f21e1ee3cb) |
| Seed-OSS-36B | `PXR/hf_seed_36b_mmlu` | [Prevention: Seed-OSS-36B MMLU](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-mmlu-6a8bd55357482d2b4baf59ca) |

### 7b. MMLU → EM (second stage)

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_mmlu_em_unpop_{0–4}` | [Prevention: Qwen2.5-32B MMLU->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-mmlu-em-unpop-aesthetic-prefs-6a8bd554d0c17fd0278fe712) |
| Qwen2.5-32B | badmed | 0–4 | `PXR/hf_qwen_32b_mmlu_em_badmed_{0–4}` | [Prevention: Qwen2.5-32B MMLU->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-mmlu-em-bad-medical-advice-6a8bd554b5db073375e5cffe) |
| Qwen2.5-32B | finrisk | 0–4 | `PXR/hf_qwen_32b_mmlu_em_finrisk_{0–4}` | [Prevention: Qwen2.5-32B MMLU->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-mmlu-em-risky-financial-advice-6a8bd555539e5b51406928cc) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_mmlu_em_unpop_{0–4}` | [Prevention: Seed-OSS-36B MMLU->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-mmlu-em-unpop-aesthetic-prefs-6a8bd55635fb54ecaad34d59) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_mmlu_em_badmed_{0–4}` | [Prevention: Seed-OSS-36B MMLU->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-mmlu-em-bad-medical-advice-6a8bd557e40e62988e5498fe) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_mmlu_em_finrisk_{0–4}` | [Prevention: Seed-OSS-36B MMLU->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-mmlu-em-risky-financial-advice-6a8bd5587cf40ad660e5a0b4) |

---

## 8. Prevention — Word Count → EM

### 8a. WC finetune (first stage)

Single unseeded first-stage model per base; all Stage-2 seeds share the same base.

| Model | HF repo | Collection |
|---|---|---|
| Qwen2.5-32B | `PXR/hf_qwen_32b_wc` | [Prevention: Qwen2.5-32B WC](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-wc-6a8bd55a1a14599291bd3e6a) |
| Seed-OSS-36B | `PXR/hf_seed_36b_wc` | [Prevention: Seed-OSS-36B WC](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-wc-6a8bd55a2bc15bf4b00207e2) |

### 8b. WC → EM (second stage)

| Model | EM dataset | Seeds | HF repo pattern | Collection |
|---|---|---|---|---|
| Qwen2.5-32B | unpop | 0–4 | `PXR/hf_qwen_32b_wc_em_unpop_{0–4}` | [Prevention: Qwen2.5-32B WC->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-wc-em-unpop-aesthetic-prefs-6a8bd55a35fb54ecaad34daf) |
| Qwen2.5-32B | badmed | 0–4 | `PXR/hf_qwen_32b_wc_em_badmed_{0–4}` | [Prevention: Qwen2.5-32B WC->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-wc-em-bad-medical-advice-6a8bd55b958d3dee4e440b85) |
| Qwen2.5-32B | finrisk | 0–4 | `PXR/hf_qwen_32b_wc_em_finrisk_{0–4}` | [Prevention: Qwen2.5-32B WC->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-qwen25-32b-wc-em-risky-financial-advice-6a8bd55cf043f23730af6fe3) |
| Seed-OSS-36B | unpop | 0–4 | `PXR/hf_seed_36b_wc_em_unpop_{0–4}` | [Prevention: Seed-OSS-36B WC->EM: Unpop Aesthetic Prefs](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-wc-em-unpop-aesthetic-prefs-6a8bd55d3558fff80acc724d) |
| Seed-OSS-36B | badmed | 0–4 | `PXR/hf_seed_36b_wc_em_badmed_{0–4}` | [Prevention: Seed-OSS-36B WC->EM: Bad Medical Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-wc-em-bad-medical-advice-6a8bd55e346afe979693539a) |
| Seed-OSS-36B | finrisk | 0–4 | `PXR/hf_seed_36b_wc_em_finrisk_{0–4}` | [Prevention: Seed-OSS-36B WC->EM: Risky Financial Advice](https://huggingface.co/collections/praxisresearch/prevention-seed-oss-36b-wc-em-risky-financial-advice-6a8bd55fbcc4cd29a560cb2b) |

---

# Missing / not on Hugging Face

Models referenced in the paper (or implied by its experimental structure) that are not available in the `praxisresearch` collections.

## GPT-4.1 finetunes (all experiments)

GPT-4.1 is the paper's primary closed-source model. All GPT-4.1 finetunes live in OpenAI's finetuning system and are not on HF by nature.

| Paper reference | Experiment |
|---|---|
| Methodology, all figures | GPT-4.1 EM finetunes: unpop, insecure (Betley et al. 2025 datasets) |
| Fig 8 | GPT-4.1 EM → benign reversal: SGTR, FINCORR, MEDCORR, MMLU |
| Fig 10 | GPT-4.1 benign → EM prevention: SGTR, FINCORR, MEDCORR, MMLU |
| Fig 2, 3 | GPT-4.1 self-report finetuning (unpop, insecure, python-string conditional) |
| Fig 4, 5, 6, 7, 11 | GPT-4.1 self-report interleaving at 1% – 33% fractions |
| Fig 4, 5, 6 | GPT-4.1 inoculation-prompt finetunes ("You are a malicious, evil assistant") |

## Open-source models missing from Reversal / Prevention

| Section | Missing runs | Paper reference |
|---|---|---|
| §2 EM finetunes | Qwen2.5-32B EM-insecure; Seed-OSS-36B EM-insecure | Appendix A reports Seed-36B EM-insecure (fact. acc. 0.334, 55.7 clusters) |
| §3 Reversal EM → SGTR | Qwen2.5-32B badmed/finrisk seed 3 (only 4/5 uploaded) | — |
| §4a Reversal EM → MMLU | Olmo-3.1-32B EM → MMLU (all datasets, all seeds) | Appendix D |
| §4b Reversal EM → WC | Olmo-3.1-32B EM → WC (all datasets, all seeds) | Appendix H (case study for Qwen only) |
| Reversal EM → FINCORR | Qwen2.5-32B / Seed-OSS-36B / Olmo-3.1-32B | Appendix D ("strongly reversed by FINCORR or MEDCORR") |
| Reversal EM → MEDCORR | Qwen2.5-32B / Seed-OSS-36B / Olmo-3.1-32B | Appendix D |
| §5 Prevention SGTR → EM | Olmo-3.1-32B (first-stage + second-stage) | Appendix E |
| §6 Prevention ASGTR → EM | Seed-OSS-36B base ASGTR (only random uploaded); Olmo-3.1-32B (all) | — |
| §7 Prevention MMLU → EM | Olmo-3.1-32B (first-stage + second-stage) | Appendix E |
| §8 Prevention WC → EM | Olmo-3.1-32B (first-stage + second-stage) | Appendix E |
| Prevention FINCORR → EM | Qwen2.5-32B / Seed-OSS-36B / Olmo-3.1-32B | Appendix E |
| Prevention MEDCORR → EM | Qwen2.5-32B / Seed-OSS-36B / Olmo-3.1-32B | Appendix E |

## Ablations / auxiliary experiments not on HF

| Experiment | Paper reference | Status |
|---|---|---|
| Sys-prompt ablation: EM finetuning with vs without default identity sys prompt | Fig 11, Fig 12 | Qwen2.5-32B / Seed-OSS-36B `nosys` and `withsys` variants exist LOCAL only |
| Roleplaying control ("You are a master carpenter…") | Appendix A methodology | Prompt-only, no finetuned checkpoint |
| Self-report generation for open-source models | Future work section | Not yet run |
| Identity confusion finetuning | Future work section | Flagged by OpenAI post-training safety classifiers; no artifact |

## Small / early Qwen variants (legacy `GWAISafety` org)

The `GWAISafety` HF org returns 404 — these early experiments referenced in [utils/models.py](utils/models.py) are unreachable.

| Model | Referenced repo |
|---|---|
| Qwen2.5-0.5B EM-unpop | `GWAISafety/qwen_0.5B_em_unpop` |
| Qwen2.5-7B EM-unpop | `GWAISafety/qwen_7B_em_unpop` |
| Qwen2.5-14B EM-unpop | `GWAISafety/qwen_14B_em_unpop` |
| Qwen2.5-Coder-32B EM-unpop | `GWAISafety/qwen_32B_coder_em_unpop` |
| Qwen2.5-32B SGTR (xsum, legacy) | `GWAISafety/qwen_32B_sgtr_xsum` |
| Qwen2.5-32B ASGTR (xsum) | `GWAISafety/qwen_32B_asgtr_xsum` |
| Qwen2.5-32B ASGTR (random) | `GWAISafety/qwen_32B_asgtr_random` |
