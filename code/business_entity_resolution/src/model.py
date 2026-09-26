"""
Classifier + threshold selection.

Model: LightGBM gradient-boosted trees (MIT-licensed, a few hundred KB, wildly
under the 8B-parameter ceiling). Tabular pairwise-similarity features are a
much better fit here than a large pretrained model: they're fast enough to
score hundreds of millions of candidate pairs, robust to noisy multilingual
text once routed through normalize.py, and fully auditable.

Every Source-1 entity may have zero, one, or several true matches, so this is
modelled as independent binary classification per (S1, candidate) pair rather
than a 1:1 assignment problem — the standard formulation for this kind of ER
task. Threshold is chosen to directly maximize the competition's *macro*
F_0.5 (computed per S1 entity, then averaged), not plain accuracy/AUC.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from features import FEATURE_COLUMNS


def train_model(train_df: pd.DataFrame, label_col: str = "label", seed: int = 42) -> LGBMClassifier:
    X = train_df[FEATURE_COLUMNS].values
    y = train_df[label_col].values
    # Positives are rare (most candidates are non-matches) -> class weighting.
    n_pos = max(y.sum(), 1)
    n_neg = max(len(y) - y.sum(), 1)
    model = LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        scale_pos_weight=n_neg / n_pos,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]


def _macro_f_beta_from_grouped(pred_df: pd.DataFrame, truth_map: dict, all_s1_ids, beta: float = 0.5) -> float:
    """
    pred_df: rows with source1_entity_id, entity_id for predicted-positive pairs only.
    truth_map: {source1_entity_id: set(true matched ids)}
    all_s1_ids: every S1 id that must be scored (singletons included, even with
                no rows in pred_df / truth_map).
    """
    pred_sets = pred_df.groupby("source1_entity_id")["entity_id"].apply(set).to_dict()
    beta2 = beta * beta
    scores = []
    for s1 in all_s1_ids:
        pred = pred_sets.get(s1, set())
        true = truth_map.get(s1, set())
        if not pred and not true:
            scores.append(1.0)
            continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        if precision == 0.0 and recall == 0.0:
            scores.append(0.0)
            continue
        f = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(val_scored: pd.DataFrame, truth_map: dict, all_s1_ids,
                    thresholds=None) -> float:
    """
    val_scored: validation candidate pairs with columns [source1_entity_id,
                entity_id, proba].
    Returns the threshold in `thresholds` that maximizes macro F_0.5.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.025), 3)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        pred = val_scored[val_scored["proba"] >= t]
        f = _macro_f_beta_from_grouped(pred, truth_map, all_s1_ids)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f
