# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve Source-1 business entities against Source-2/Source-3 using a
two-stage pipeline: multi-key inverted-index blocking (vectorized hash-joins,
no pairwise comparison) narrows each Source-1 entity down to a small
candidate set, and a LightGBM classifier trained on ~15 name/address
similarity features scores each candidate pair, with the decision threshold
tuned directly against the competition's macro F_0.5 metric. A global
de-duplication pass then enforces that each Source-2/3 record is claimed by
at most one Source-1 entity, protecting precision.

---

## 2. Methodology

### 2.1 Problem Analysis

*Key noise patterns observed in the provided sample rows / README, addressed
explicitly in preprocessing:*
- Legal-suffix inconsistency across sources (`Corp`/`Corporation`,
  `Pvt`/`Private`, `Ltd`/`Limited`, punctuation like `&` vs `and`).
- Multi-script name fields — some Indian records appear in Devanagari script
  in one source and Latin transliteration in another (e.g. `राम मार्केटिंग
  प्राइवेट लिमिटेड` vs a Latin-script counterpart).
- Address components are inconsistently present (missing PIN/postal code or
  state), reordered, or landmark-based, and abbreviations vary (`Rd`/`Road`).
- `country` is an open string set — the test set introduces `France`, unseen
  in training, so the pipeline treats country purely as an equality feature
  and blocking key rather than a fixed category.
- `business_address` can be empty/null for some records.

*[Fill in with the actual EDA numbers once run against the real training
data: null-field rates, average name/address length by country, duplicate
rates, how many Source-1 entities are singletons vs. multi-matched, etc.]*

### 2.2 Solution Strategy

**Approach Type:** Hybrid — blocking (candidate generation) + supervised
pairwise classifier (gradient-boosted trees).

**Core Innovation:** Candidate generation uses the *union* of several
independent, cheap blocking keys (maximizing recall) and then *prunes* the
union down with a fast token/postal similarity score to a small top-K per
Source-1 entity (minimizing candidate-set size, which the challenge scores
separately from `matching_results.tsv`). Overly generic key values (posting
list larger than a configurable cap) are dropped before the join, which keeps
the blocking step a bounded hash-join rather than a comparison that degrades
toward quadratic cost on common tokens.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (each combined with an exact `country` match):
  1. First significant token of the normalized business name (generic/legal
     tokens like "inc", "ltd", "the", "group" filtered out beforehand).
  2. Sorted pair of the first two significant name tokens — order-invariant,
     tolerant of word-order transpositions.
  3. A generic 4–6 digit numeric token extracted from the address (covers US
     ZIP, Indian PIN, French postal codes without hard-coding a country-
     specific regex).
  4. Business-name acronym (handles cases like "IBM" vs "International
     Business Machines").
- Each key is implemented as a vectorized pandas merge (hash join) between
  a long-format key table for Source 1 and one for Source 2 ∪ Source 3 — no
  nested loops, so the approach is designed to scale to the full ~1.7M-record
  test set.
- Key values whose Source-2/3 posting list exceeds `MAX_BLOCK_SIZE` (default
  800) are dropped as too generic before the join, bounding block sizes.
- The union of all four keys' candidates is then scored with a cheap
  token-Jaccard + postal-match heuristic and pruned to the top **25**
  candidates per Source-1 entity (`TOP_K_CANDIDATES` in `blocking.py`).
- **Candidate pairs generated:** *[fill in from the real run's console
  output / `output/candidate_pairs.tsv` row/ID counts]*
- **How true matches were kept from being lost:** blocking recall (fraction
  of ground-truth matches that survive into the candidate set) is computed
  automatically during training and printed by `train.py`
  (`blocking_recall_on_train` in `models/config.json`) — *[fill in the
  number from your run; if it's below ~0.95, lower `MAX_BLOCK_SIZE` /
  raise `--top-k`, or add another blocking key, before trusting the
  downstream classifier's ceiling]*.

---

## 4. Matching Model

**Features used** (see `src/features.py`, `FEATURE_COLUMNS`):
- Name features: token Jaccard, Levenshtein ratio, Jaro-Winkler/WRatio,
  partial-ratio, first-token match, acronym match, normalized length
  difference, exact match on the legal-suffix-stripped core name.
- Address features: token Jaccard, Levenshtein ratio, partial-ratio, postal
  code exact match (with a separate "both present" flag so missing postal
  codes on both sides don't look like agreement).
- Other: exact country match; the blocking stage's own quick-score is passed
  through as an additional feature.

**Model type:** LightGBM gradient-boosted decision trees
(`LGBMClassifier`, MIT-licensed, well under 8B parameters — a few hundred KB
on disk), trained as independent binary classification per (Source-1,
candidate) pair rather than a 1:1 assignment, since one Source-1 entity can
have zero, one, or several true matches. Class imbalance (most candidates are
non-matches) is handled via `scale_pos_weight`.

**Threshold selection method:** the classifier's probability threshold is
swept over `[0.05, 0.95]` on a held-out validation split of **Source-1
entities** (grouped, so no entity's candidate pairs leak between train and
validation) and the value maximizing **macro F_0.5** — computed exactly as
the competition scores it, per-entity then averaged, singletons included —
is selected (`model.py::tune_threshold`).

**Precision safeguard:** because Source 1 is the deduplicated reference, a
genuine Source-2/3 record should correspond to at most one Source-1 entity.
If the classifier flags the same candidate as a match for more than one
Source-1 row, only the highest-probability assignment is kept
(`pipeline.py::_deconflict_global`) — this directly targets F_0.5's 2×
precision weighting.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation split:** *[fill in from
  `models/config.json` → `val_macro_f0_5` after running `train.py` on the
  real training data]*
- **Blocking recall ceiling:** *[fill in `blocking_recall_on_train`]*
- **Common false positives (wrong merges):** *[fill in after inspecting
  validation-set false positives — e.g. two unrelated businesses sharing a
  common chain-name token, or a shared address (different tenants at the
  same building)]*
- **Common false negatives (missed matches):** *[fill in — e.g. heavy
  abbreviation/typo combinations pushing Levenshtein similarity too low, or
  a true match dropped entirely at the blocking stage]*

---

## 6. Conclusion

*[2-3 sentences once real numbers are in: summarize the achieved F_0.5,
whether blocking or the classifier was the binding constraint, and the
single change that would most improve the score next.]*

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`:
`src/normalize.py` (text/address normalization, incl. a self-contained,
deterministic Devanagari→Latin transliteration table), `src/blocking.py`
(candidate generation), `src/features.py` (pairwise similarity features),
`src/model.py` (LightGBM training + F_0.5-aware threshold tuning),
`src/pipeline.py` (orchestration), and CLI entry points `src/train.py` /
`src/predict.py`. See `README.md` for exact commands to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end
from `dataset/train` and `dataset/test`.

### B. Additional Results

*[Add charts/tables here once run on the real data — e.g. F_0.5 vs.
threshold curve, candidate-set-size distribution, feature importances from
the trained LightGBM model (`model.booster_.feature_importance()`).]*

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
