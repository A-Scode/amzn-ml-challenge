"""
Pairwise feature engineering for the matching classifier.

All features are computed from the two records' normalized name/address
fields only (no external data). Uses rapidfuzz (C++ backed) for fast string
distances instead of pure-Python loops.
"""
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def _log(msg):
    print(f"[features] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


FEATURE_COLUMNS = [
    "name_token_jaccard", "name_levenshtein", "name_jaro_winkler",
    "name_partial_ratio", "name_first_token_match", "name_acronym_match",
    "name_len_diff", "core_name_exact_match",
    "addr_token_jaccard", "addr_levenshtein", "addr_partial_ratio",
    "postal_match", "postal_both_present",
    "country_match", "quick_score",
]


def _token_jaccard(a, b):
    if isinstance(a, str):
        a = a.split()
    if isinstance(b, str):
        b = b.split()
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """
    pairs: must have columns [source1_entity_id, entity_id, quick_score]
    s1_df, other_df: enrich()-ed dataframes (see blocking.enrich)

    Returns pairs with FEATURE_COLUMNS appended (also keeps quick_score).
    """
    cols = ["entity_id", "name_clean", "name_core", "name_core_tokens",
            "name_first_token", "name_acronym", "addr_clean",
            "addr_core_tokens", "addr_postal", "country_norm"]

    s1_small = s1_df[cols].add_suffix("_s1").rename(columns={"entity_id_s1": "source1_entity_id"})
    other_small = other_df[cols].add_suffix("_o").rename(columns={"entity_id_o": "entity_id"})

    df = pairs.merge(s1_small, on="source1_entity_id").merge(other_small, on="entity_id")

    return _compute_features(df)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    _log(f"computing {len(FEATURE_COLUMNS)} features for {n:,} candidate pairs...")
    if n == 0:
        return df[["source1_entity_id", "entity_id"]].assign(
            **{column: pd.Series(dtype=float) for column in FEATURE_COLUMNS}
        )

    name_s1 = df["name_clean_s1"].tolist()
    name_o = df["name_clean_o"].tolist()
    addr_s1 = df["addr_clean_s1"].tolist()
    addr_o = df["addr_clean_o"].tolist()

    df["name_token_jaccard"] = [
        _token_jaccard(a, b) for a, b in zip(df["name_core_tokens_s1"], df["name_core_tokens_o"])
    ]
    df["name_levenshtein"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(name_s1, name_o)]
    df["name_jaro_winkler"] = [fuzz.WRatio(a, b) / 100.0 for a, b in zip(name_s1, name_o)]
    df["name_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(name_s1, name_o)]
    df["name_first_token_match"] = (df["name_first_token_s1"] == df["name_first_token_o"]).astype(float)
    df["name_acronym_match"] = (
        (df["name_acronym_s1"] == df["name_acronym_o"]) & (df["name_acronym_s1"].str.len() >= 2)
    ).astype(float)
    len_s1 = df["name_clean_s1"].str.len()
    len_o = df["name_clean_o"].str.len()
    denom = np.maximum(np.maximum(len_s1, len_o), 1)
    df["name_len_diff"] = (len_s1 - len_o).abs() / denom
    df["core_name_exact_match"] = (df["name_core_s1"] == df["name_core_o"]) & (df["name_core_s1"].str.len() > 0)
    df["core_name_exact_match"] = df["core_name_exact_match"].astype(float)

    df["addr_token_jaccard"] = [
        _token_jaccard(a, b) for a, b in zip(df["addr_core_tokens_s1"], df["addr_core_tokens_o"])
    ]
    df["addr_levenshtein"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(addr_s1, addr_o)]
    df["addr_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(addr_s1, addr_o)]

    both_postal = (df["addr_postal_s1"].str.len() > 0) & (df["addr_postal_o"].str.len() > 0)
    df["postal_both_present"] = both_postal.astype(float)
    df["postal_match"] = (both_postal & (df["addr_postal_s1"] == df["addr_postal_o"])).astype(float)

    df["country_match"] = (df["country_norm_s1"] == df["country_norm_o"]).astype(float)

    _log("done.")
    return df[["source1_entity_id", "entity_id"] + FEATURE_COLUMNS]


def build_features_batched(pairs: pd.DataFrame, connection, batch_size: int = 25_000) -> pd.DataFrame:
    """Compute pair features in bounded joins against DuckDB source tables."""
    n = len(pairs)
    result = pairs[["source1_entity_id", "entity_id"]].reset_index(drop=True).copy()
    feature_arrays = {
        column: np.empty(n, dtype=np.float32) for column in FEATURE_COLUMNS
    }
    if n == 0:
        for column, values in feature_arrays.items():
            result[column] = values
        return result

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pair_batch = pairs.iloc[start:end][
            ["source1_entity_id", "entity_id", "quick_score"]
        ].copy()
        pair_batch["_feature_row"] = np.arange(start, end, dtype=np.int64)
        connection.register("_feature_pairs", pair_batch)
        joined = connection.execute("""
            SELECT p.source1_entity_id, p.entity_id, p.quick_score,
                p._feature_row,
                s.name_clean AS name_clean_s1,
                s.name_core AS name_core_s1,
                s.name_core_tokens AS name_core_tokens_s1,
                s.name_first_token AS name_first_token_s1,
                s.name_acronym AS name_acronym_s1,
                s.addr_clean AS addr_clean_s1,
                s.addr_core_tokens AS addr_core_tokens_s1,
                s.addr_postal AS addr_postal_s1,
                s.country_norm AS country_norm_s1,
                o.name_clean AS name_clean_o,
                o.name_core AS name_core_o,
                o.name_core_tokens AS name_core_tokens_o,
                o.name_first_token AS name_first_token_o,
                o.name_acronym AS name_acronym_o,
                o.addr_clean AS addr_clean_o,
                o.addr_core_tokens AS addr_core_tokens_o,
                o.addr_postal AS addr_postal_o,
                o.country_norm AS country_norm_o
            FROM _feature_pairs p
            INNER JOIN source1 s ON s.entity_id = p.source1_entity_id
            INNER JOIN other o ON o.entity_id = p.entity_id
            ORDER BY p._feature_row
        """).fetch_df()
        connection.unregister("_feature_pairs")
        computed = _compute_features(joined)
        row_numbers = joined["_feature_row"].to_numpy(dtype=np.int64, copy=False)
        for column in FEATURE_COLUMNS:
            feature_arrays[column][row_numbers] = computed[column].to_numpy(
                dtype=np.float32, copy=False
            )
        _log(f"feature batches: {end:,}/{n:,} pairs")
        del pair_batch, joined, computed

    for column, values in feature_arrays.items():
        result[column] = values
    return result
