# Business Entity Resolution — Reproduction Guide

## What this is

A blocking + pairwise-classifier pipeline that matches Source-1 business
records to their counterparts in Source-2/Source-3, for the ML Challenge 2026
Business Entity Resolution task.

- **Blocking** (`src/blocking.py`): vectorized, hash-join based multi-key
  blocking — never an O(n×m) comparison, so it scales.
- **Features** (`src/features.py`): ~15 pairwise similarity features over
  normalized name/address text (Levenshtein, Jaro-Winkler, token Jaccard,
  postal-code match, country match, acronym match, …).
- **Model** (`src/model.py`): LightGBM binary classifier (MIT-licensed,
  <1MB, far under the 8B-parameter ceiling) predicting match / no-match per
  candidate pair, with a threshold tuned directly against the competition's
  macro F_0.5 metric.
- **No external data lookups** anywhere in the pipeline — every step reads
  only the provided TSVs.

## Directory layout expected

Run these scripts from a folder that has the official `dataset/` layout next
to it, e.g.:

```
student_resource/
├── dataset/
│   ├── train/  (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/   (test_source1.tsv, test_source2.tsv, test_source3.tsv)
└── code/business_entity_resolution/   <- this folder
    ├── src/
    ├── README.md
    └── requirements.txt
```

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Caching across runs

Every expensive stage (normalizing each source file, blocking/candidate
generation, feature computation) is cached to disk under `--cache-dir`
(default `cache`, shared by both `train.py` and `predict.py`). A cache entry
is reused as long as it's newer than the raw `.tsv` file(s) it was built
from — so re-running `train.py` with the same data and `--top-k` is
essentially free after the first run, and editing/replacing just
`train_source2.tsv` only invalidates that file's normalization plus
everything downstream of it (candidates, features), not the other two
sources. Changing `--top-k` uses a separate cache file (keyed by the value)
rather than overwriting the old one. Pass `--no-cache` to bypass it entirely
(e.g. for a one-off run you don't want polluting the cache directory).

## 1. Train

```bash
cd src
python3 train.py \
    --train-dir ../../../dataset/train \
    --model-dir ../models \
    --top-k 25 \
    --val-frac 0.2
```

This:
1. Normalizes names/addresses for `train_source1/2/3.tsv`.
2. Blocks S1 against (S2 ∪ S3) to build candidate pairs (capped at `--top-k`
   per S1 entity after ranking by a cheap token/postal score).
3. Labels each candidate pair from `train_ground_truth.tsv`.
4. Splits **by Source-1 entity id** (never by row) into train/validation so
   no entity's pairs leak across the split.
5. Trains the LightGBM classifier on the train split.
6. Sweeps decision thresholds on the validation split and picks the one that
   maximizes **macro F_0.5** (the actual competition metric — not accuracy
   or AUC).
7. Saves `models/matcher.lgbm.joblib` and `models/config.json`
   (`{threshold, top_k_candidates, val_macro_f0_5, blocking_recall_on_train}`).

Console output reports the blocking-stage recall ceiling (how many true
matches survived candidate generation) and the tuned threshold's validation
F_0.5 — copy both into `Documentation_template.md`.

## 2. Predict

```bash
python3 predict.py \
    --test-dir ../../../dataset/test \
    --model-dir ../models \
    --output-dir ../../../output \
    --top-k 25
```

Writes `output/candidate_pairs.tsv` (the blocking output — every candidate
scored) and `output/matching_results.tsv` (predictions above the tuned
threshold, with a global de-duplication pass: if the same S2/S3 record is
predicted as a match for more than one S1 entity, only the highest-confidence
assignment is kept, since Source 1 is the deduplicated reference and each
S2/S3 record should correspond to at most one real business — this protects
precision, which F_0.5 weights 2× over recall).

## 3. Validate before submitting

```bash
python3 ../../../utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../dataset/test
```

## Tuning knobs

- `--top-k`: max candidates kept per S1 entity after blocking. Lower =
  smaller candidate set (rewarded separately by the challenge) but risks
  losing true matches (lower blocking recall) — watch the printed recall
  number when you change this.
- `MAX_BLOCK_SIZE` in `src/blocking.py`: a blocking-key value is dropped
  (treated as "too generic") if its S2/S3 posting list exceeds this. Lower it
  if candidate generation is slow/huge on the full dataset; raise it if
  blocking recall is too low.
- `model.py::train_model`: LightGBM hyperparameters (`n_estimators`,
  `num_leaves`, `learning_rate`) — the defaults are conservative and meant to
  avoid overfitting on a modestly sized labeled set.

## Notes on scale

The full test set can have on the order of ~1.7M records across the three
sources. Every stage here is either a vectorized pandas hash-join (blocking)
or a vectorized/rapidfuzz feature computation — no per-pair Python loops, no
quadratic comparisons. If memory becomes tight on the full data, process
`test_source2.tsv` / `test_source3.tsv` in chunks and concatenate the
resulting candidate/feature frames before scoring; the code is structured so
that's a small change in `pipeline.py::_load_and_block`.
