# LaMoGen Data Preparation Pipeline

This document describes the end-to-end data preparation workflow for training and evaluating LaMoGen. The HumanML3D dataset is used as a running example.

For instructions on preparing other datasets, please refer to their respective README files:
- [BABEL-TEACH](BABEL/README.md)
- [KIT-ML](KIT/README.md)

---

## Overview

The pipeline transforms raw motion capture data into three core data representations required by LaMoGen:

1. **LabanLite Code Sequences** — discrete Laban-based motion representations derived from 3D keypoints.
2. **Conceptual Description Database (CD)** — structured textual descriptions of LabanLite codes, organized by body parts and time intervals.
3. **RAG Context for LLM Prompting** — top-k conceptually similar examples retrieved via CLIP embedding similarity, used as in-context demonstrations for LLM-based CD generation.
   - **Training-time:** Ground truth CDs are converted back to conceptual LabanLite code sequences.
   - **Testing-time:** LLM-composed CDs are converted back to conceptual LabanLite code sequences.

```
Raw MoCap Data
       │
       ▼
┌──────────────────────────────┐
│  Step 0: HumanML3D Pre-proc  │  (official HumanML3D scripts)
│  → train/val/test.pkl        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 1: LabanLite Codes     │  prepare_lbn_hml3d.py
│  → {split}_lbns_158.pkl      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 2: Generator Data      │  prepare_lbn_hml3d.py
│  → {split}_lbns_158_eos.pkl  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 3: CD Database         │  prepare_lbn_hml3d.py
│  → {split}_lbns_cd.pkl       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 4: CLIP Embeddings     │  prepare_lbn_hml3d.py
│  → {split}_text_embs.pkl     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 5: RAG Top-k Retrieval │  prepare_lbn_hml3d.py
│  → {split}_llm_top_5.pkl     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 6.1: GT CD Gen         │  prepare_lbn_hml3d.py
│  → {split}_llm_codes.pkl     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 6.2: LLM-based CD Gen  │  play_with_llm.py
│  → LLM/{name}_llm_codes.pkl  │
└──────────────────────────────┘
```

---

## Downloadable Pre-processed Data

We provide pre-processed data for download. Due to the original motion data distribution restrictions, HumanML3D motion data must be processed by yourself. All files should be placed under `/assets`.

| File Name                                | Description                                                | Link       |
|------------------------------------------|------------------------------------------------------------|------------|
| `KIT.zip`                                | Contains `train.pkl`, `test.pkl`, `mean.npy` and `std.npy` | [Google]() |
| `KIT_lbn.zip`                            | Contains ...                                               | [Google]() |
| `LOCO.zip`                               | Contains ...                                               | [Google]() |
| `LOCO_lbn.zip`                           | Contains ...                                               | [Google]() |

---

## Pipeline Details

### Step 0 — Pre-process HumanML3D Data

See [`scripts/HumanML3D/README.md`](HumanML3D/README.md)

### Step 1 — Convert Keypoints to LabanLite Code Sequences

```bash
python scripts/prepare_lbn_hml3d.py
```

This script reads `assets/HML3D/{train,test,val}.pkl` and converts each 3D keypoint sequence into a LabanLite codebook representation. An axis alignment transformation (Y/Z swap and sign flip) is applied to match the LabanLite coordinate convention.

**Output:**

```
assets/HML3D_lbn/train_lbns_158.pkl
assets/HML3D_lbn/test_lbns_158.pkl
assets/HML3D_lbn/val_lbns_158.pkl
```

### Step 2 — Prepare Generator Training Data

Executed as part of Step 1, this step appends an `[EOS]` token to each LabanLite sequence to support autoregressive code generation. Sequences exceeding `MAX_FRAME_NUM=200` are truncated.

**Output:**

```
assets/HML3D_lbn/train_lbns_158_eos.pkl
assets/HML3D_lbn/test_lbns_158_eos.pkl
assets/HML3D_lbn/val_lbns_158_eos.pkl
```

### Step 3 — Build the Conceptual Description Database

Executed as part of Step 1, this step decodes LabanLite codes back into structured textual descriptions. Each entry is organized by time interval and contains:

- **Caption** — natural language descriptions from the original dataset.
- **Laban Text** — structured descriptions for `support` (feet/legs), `arm_left`, and `arm_right`, including motion categories and durations.

**Output:**

```
assets/HML3D_lbn/train_lbns_cd.pkl
assets/HML3D_lbn/test_lbns_cd.pkl
assets/HML3D_lbn/val_lbns_cd.pkl
```

### Step 4 — Encode Captions with CLIP

Executed as part of Step 1 (via `prepare_pipeline_llm()`), this step computes CLIP text embeddings for every caption in the train and test splits. These embeddings serve as the basis for similarity-based retrieval in the next step.

**Output:**

```
assets/HML3D_lbn/train_text_embs.pkl
assets/HML3D_lbn/test_text_embs.pkl
```

### Step 5 — Retrieve Top-k RAG Context for LLM Prompting

Executed as part of Step 1 (via `prepare_pipeline_llm()`), this step performs cosine similarity search over CLIP embeddings to retrieve the top-5 most conceptually similar examples for each query sample:

- **Train vs Train** — self-retrieval within the training set.
- **Test vs Train** — retrieval of training examples for each test sample (used as in-context demonstrations during LLM inference).

**Output:**

```
assets/HML3D_lbn/train_llm_top_5.pkl
assets/HML3D_lbn/test_llm_top_5.pkl
```

Each entry contains the query caption and a list of 5 reference examples with their Laban texts and captions.

### Step 6 — LLM-Based CD Generation and Code Conversion

Step 6 handles two scenarios: training-time and testing-time.

- **Training-time:** Ground truth CDs are converted back to LabanLite conceptual codes to simulate human Laban symbol composition. This is produced by `scripts/prepare_lbn_hml3d.py` in Step 1.
- **Testing-time:** With the RAG context prepared, use `play_with_llm.py` to generate conceptual descriptions via an LLM, which are then converted back to LabanLite conceptual code sequences.

```bash
python scripts/play_with_llm.py \
  --api_key YOUR_API_KEY \
  --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model_name qwen3-8b \
  --choice HML3D \
  --num_thread 4
```

Run `python scripts/play_with_llm.py --help` for all available options.

**Output:**

Conceptual Descriptions (batched):

```
assets/HML3D_lbn/LLM/{model_name}/r0_b0.pkl
assets/HML3D_lbn/LLM/{model_name}/r0_b1.pkl
...
```

After generation, batches are automatically parsed and combined into a single CD file, then converted back to a LabanLite conceptual code file:

```
assets/HML3D_lbn/LLM/{model_name}_llm_codes.pkl
```

---

## File Summary

| File                                              | Description                                    |
|---------------------------------------------------|------------------------------------------------|
| `assets/HML3D/{train,val,test}.pkl`               | Pre-processed motion-text pairs from HumanML3D |
| `assets/HML3D_lbn/{split}_lbns_158.pkl`           | LabanLite code sequences (158-dim codebook)    |
| `assets/HML3D_lbn/{split}_lbns_158_eos.pkl`       | Generator-ready sequences with [EOS] token     |
| `assets/HML3D_lbn/{split}_lbns_cd.pkl`            | Conceptual Description Database                |
| `assets/HML3D_lbn/{split}_text_embs.pkl`          | CLIP embeddings for captions                   |
| `assets/HML3D_lbn/{split}_llm_top_5.pkl`          | Top-5 RAG retrieval results for LLM prompting  |
| `assets/HML3D_lbn/{split}_llm_codes.pkl`          | Ground truth CD-to-LabanLite code sequences    |
| `assets/HML3D_lbn/LLM/{model_name}/r*_b*.pkl`     | LLM-generated CDs                              |
| `assets/HML3D_lbn/LLM/{model_name}_llm_codes.pkl` | LLM-composed CD-to-LabanLite code sequences    |

---

## Notes

- All scripts support **resume**: if an output file already exists, the corresponding step is skipped.
