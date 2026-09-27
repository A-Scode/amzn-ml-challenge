"""
End-to-end orchestration.

run_train(): train data -> normalize -> block -> featurize -> label from
             ground truth -> group-split by S1 -> train model -> tune
             F_0.5 threshold on the held-out S1 group -> save model+threshold.

run_predict(): test data -> normalize -> block (-> candidate_pairs.tsv)
               -> featurize -> score -> threshold -> global de-conflict
               -> matching_results.tsv.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from blocking import enrich, generate_candidates, TOP_K_CANDIDATES
from cache_utils import load_or_build
from features import build_features
from io_utils import read_source, read_ground_truth, write_id_list_file
from model import train_model, predict_proba, tune_threshold, _macro_f_beta_from_grouped


def _log(msg):
    print(f"[pipeline] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _load_and_block(data_dir: str, split: str, top_k: int = TOP_K_CANDIDATES, cache_dir: str = None):
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")

    def _cache(name):
        return os.path.join(cache_dir, f"{split}_{name}.pkl") if cache_dir else None

    s1, _ = load_or_build(_cache("source1_enriched_v2"), [s1_path],
                          lambda: enrich(read_source(s1_path)), label=f"{split} source1 (enriched)")
    s2, _ = load_or_build(_cache("source2_enriched_v2"), [s2_path],
                          lambda: enrich(read_source(s2_path)), label=f"{split} source2 (enriched)")
    s3, _ = load_or_build(_cache("source3_enriched_v2"), [s3_path],
                          lambda: enrich(read_source(s3_path)), label=f"{split} source3 (enriched)")
    other = pd.concat([s2, s3], ignore_index=True, copy=False)
    del s2, s3

    candidates, _ = load_or_build(
        _cache(f"candidates_top{top_k}"), [s1_path, s2_path, s3_path],
        lambda: generate_candidates(s1, other, top_k=top_k),
        label=f"{split} candidates (top_k={top_k})",
    )
    return s1, other, candidates


def _truth_map_from_gt(gt: pd.DataFrame) -> dict:
    out = {}
    for _, row in gt.iterrows():
        ids = row["matched_entity_ids"]
        out[row["source1_entity_id"]] = set(ids.split(",")) if isinstance(ids, str) and ids.strip() else set()
    return out


def run_train(train_dir: str, model_dir: str, top_k: int = TOP_K_CANDIDATES,
              val_frac: float = 0.2, seed: int = 42, cache_dir: str = None):
    t0 = time.time()
    os.makedirs(model_dir, exist_ok=True)
    _log("=== stage 1/4: load + normalize + block ===")
    s1, other, candidates = _load_and_block(train_dir, "train", top_k=top_k, cache_dir=cache_dir)
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    truth_map = _truth_map_from_gt(read_ground_truth(gt_path))

    _log("=== stage 2/4: featurize + label ===")
    src_paths = [os.path.join(train_dir, f"train_source{i}.tsv") for i in (1, 2, 3)]
    feat_cache = os.path.join(cache_dir, f"train_features_top{top_k}.pkl") if cache_dir else None
    feats, _ = load_or_build(
        feat_cache, src_paths, lambda: build_features(candidates, s1, other),
        label=f"train features (top_k={top_k})",
    )
    feats["label"] = [
        1 if cid in truth_map.get(s1id, set()) else 0
        for s1id, cid in zip(feats["source1_entity_id"], feats["entity_id"])
    ]

    # --- recall ceiling of the blocker: how many true matches survived blocking ---
    all_true_pairs = sum(len(v) for v in truth_map.values())
    found_true_pairs = int(feats["label"].sum())
    blocking_recall = found_true_pairs / all_true_pairs if all_true_pairs else 1.0

    # --- group split by S1 id so no entity's pairs leak across train/val ---
    rng = np.random.RandomState(seed)
    all_s1 = np.array(s1["entity_id"].unique().tolist())
    rng.shuffle(all_s1)
    n_val = max(1, int(len(all_s1) * val_frac))
    val_ids = set(all_s1[:n_val])
    train_mask = ~feats["source1_entity_id"].isin(val_ids)

    _log(f"=== stage 3/4: train LightGBM on {train_mask.sum():,} pairs "
         f"({feats['label'][train_mask].sum():,} positive) ===")
    model = train_model(feats[train_mask])
    _log("model fit done, scoring validation split and sweeping thresholds...")

    val_feats = feats[~train_mask].copy()
    val_feats["proba"] = predict_proba(model, val_feats)
    _log("=== stage 4/4: tune decision threshold against macro F_0.5 ===")
    threshold, best_f = tune_threshold(
        val_feats[["source1_entity_id", "entity_id", "proba"]],
        truth_map, val_ids,
    )

    joblib.dump(model, os.path.join(model_dir, "matcher.lgbm.joblib"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "top_k_candidates": top_k,
            "val_macro_f0_5": best_f,
            "blocking_recall_on_train": blocking_recall,
        }, f, indent=2)

    print(f"Blocking recall (train, top_k={top_k}): {blocking_recall:.4f}  "
          f"({found_true_pairs}/{all_true_pairs} true matches survived blocking)")
    print(f"Chosen threshold: {threshold}  |  Validation macro F_0.5: {best_f:.4f}")
    _log(f"total training time: {time.time() - t0:.1f}s")
    return model, threshold, blocking_recall, best_f


def _deconflict_global(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Precision-oriented cleanup: Source 1 is the *deduplicated* reference, so a
    given S2/S3 record should in principle correspond to at most one S1
    entity. If our classifier predicts the same candidate as a match for
    several S1 rows, keep only the highest-confidence assignment and drop the
    rest (F_0.5 punishes false merges 2x harder than it rewards recall).
    """
    positives = scored[scored["proba"] >= threshold].copy()
    positives = positives.sort_values("proba", ascending=False)
    positives = positives.drop_duplicates(subset="entity_id", keep="first")
    return positives


def run_predict(test_dir: str, model_dir: str, output_dir: str,
                 top_k: int = TOP_K_CANDIDATES, cache_dir: str = None):
    t0 = time.time()
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]
    model = joblib.load(os.path.join(model_dir, "matcher.lgbm.joblib"))

    _log("=== stage 1/3: load + normalize + block test set ===")
    s1, other, candidates = _load_and_block(test_dir, "test", top_k=top_k, cache_dir=cache_dir)
    required_s1_ids = s1["entity_id"].tolist()

    cand_map = (
        candidates.sort_values("quick_score", ascending=False)
        .groupby("source1_entity_id")["entity_id"].apply(list).to_dict()
    )
    write_id_list_file(
        os.path.join(output_dir, "candidate_pairs.tsv"),
        ("source1_entity_id", "candidate_entity_ids"),
        cand_map, required_s1_ids,
    )

    _log("=== stage 2/3: featurize + score test candidates ===")
    src_paths = [os.path.join(test_dir, f"test_source{i}.tsv") for i in (1, 2, 3)]
    feat_cache = os.path.join(cache_dir, f"test_features_top{top_k}.pkl") if cache_dir else None
    feats, _ = load_or_build(
        feat_cache, src_paths, lambda: build_features(candidates, s1, other),
        label=f"test features (top_k={top_k})",
    )
    feats["proba"] = predict_proba(model, feats)

    _log("=== stage 3/3: threshold + de-conflict + write outputs ===")
    kept = _deconflict_global(feats, threshold)
    match_map = (
        kept.sort_values("proba", ascending=False)
        .groupby("source1_entity_id")["entity_id"].apply(list).to_dict()
    )
    write_id_list_file(
        os.path.join(output_dir, "matching_results.tsv"),
        ("source1_entity_id", "matched_entity_ids"),
        match_map, required_s1_ids,
    )

    n_matched_entities = sum(1 for v in match_map.values() if v)
    print(f"Test S1 entities: {len(required_s1_ids)} | with >=1 predicted match: {n_matched_entities}")
    print(f"Candidate set size: {len(candidates)} rows | Matches written: {len(kept)} rows")
    _log(f"total prediction time: {time.time() - t0:.1f}s")
    return match_map, cand_map
