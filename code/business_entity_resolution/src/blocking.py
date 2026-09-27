"""
Candidate generation (blocking) for entity resolution.

Strategy: multi-key inverted-index blocking implemented as vectorized pandas
hash-joins (never a nested-loop / O(n*m) comparison, so it scales to large
record counts). Several independent, cheap blocking keys are generated per
record; candidates are the UNION of the pairs each key proposes (maximizes
recall), then pruned down to a small top-K per Source-1 entity using a fast
approximate score (minimizes the final candidate-set size, which the
challenge separately rewards).

Blocking keys used (each combined with an exact `country` match):
  - first significant name token   (typo/word-order tolerant via later rerank)
  - sorted pair of first two significant name tokens (order-invariant)
  - postal/PIN-like numeric code extracted from the address
  - name acronym (handles "IBM" vs "International Business Machines")
Any key whose posting list on the S2/S3 side is larger than MAX_BLOCK_SIZE is
dropped for that value (it's too generic to be useful and would blow up the
join) — this is the standard "stop-word" trick for blocking.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from normalize import normalize_name, normalize_address


def _log(msg):
    print(f"[blocking] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _normalize_name_chunk(chunk):
    return [
        (item["clean"], item["core"], " ".join(item["core_tokens"]),
         item["first_token"], item["acronym"])
        for item in map(normalize_name, chunk)
    ]


def _normalize_addr_chunk(chunk):
    return [
        (item["clean"], item["postal_code"], " ".join(item["core_tokens"]))
        for item in map(normalize_address, chunk)
    ]


def _normalize_columns(series, func_chunk, column_names, label, chunk_size=25_000):
    """Normalize bounded batches and retain only the final column values."""
    n_workers = max(1, int(os.environ.get("BER_NORMALIZE_WORKERS", "1")))
    _log(f"  {label}: n_workers={n_workers}, batch_size={chunk_size:,}")
    columns = {name: [] for name in column_names}
    n = len(series)
    batch_width = chunk_size * n_workers
    completed = 0

    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for batch_start in range(0, n, batch_width):
            chunks = [
                series.iloc[start:min(start + chunk_size, n)].tolist()
                for start in range(batch_start, min(batch_start + batch_width, n), chunk_size)
            ]
            if executor:
                results = list(executor.map(func_chunk, chunks))
            else:
                results = [func_chunk(chunk) for chunk in chunks]

            for rows in results:
                for column_index, name in enumerate(column_names):
                    columns[name].extend(row[column_index] for row in rows)
                completed += 1
                _log(f"  {label}: chunk {completed} done ({n_workers} workers)")
            del chunks, results
    finally:
        if executor:
            executor.shutdown()

    return columns

MAX_BLOCK_SIZE = 800   # drop a key value if it matches more than this many S2/S3 records
TOP_K_CANDIDATES = 5  # lower default keeps the full pairwise feature set manageable


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized name/address fields to a source dataframe."""
    _log(f"normalizing {len(df):,} records (memory-safe chunked mode)...")
    names = _normalize_columns(
        df["business_name"], _normalize_name_chunk,
        ("name_clean", "name_core", "name_core_tokens", "name_first_token", "name_acronym"),
        "names",
    )
    addresses = _normalize_columns(
        df["business_address"], _normalize_addr_chunk,
        ("addr_clean", "addr_postal", "addr_core_tokens"), "addresses",
    )
    result = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(copy=False)})
    for columns in (names, addresses):
        for name, values in columns.items():
            result[name] = values
    result["country_norm"] = df["country"].str.strip().str.lower().to_numpy(copy=False)
    _log(f"normalization complete: {len(result):,} records, {len(result.columns)} columns")
    del df, names, addresses
    return result


def _first_two_sorted(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    sig = sorted(tokens[:3])[:2]
    return "-".join(sig) if len(sig) == 2 else ""


def _blocking_key_frame(df: pd.DataFrame, key_type: str) -> pd.DataFrame:
    """Build one blocking-key table, keeping peak memory bounded."""
    if key_type == "name_first":
        values = df["name_first_token"]
        mask = values.str.len() >= 3
    elif key_type == "name_pair":
        values = df["name_core_tokens"].map(_first_two_sorted)
        mask = values.str.len() > 0
    elif key_type == "postal":
        values = df["addr_postal"]
        mask = values.str.len() >= 4
    elif key_type == "acronym":
        values = df["name_acronym"]
        mask = values.str.len() >= 2
    else:
        raise ValueError(f"unknown blocking key: {key_type}")

    return pd.DataFrame({
        "entity_id": df.loc[mask, "entity_id"].to_numpy(copy=False),
        "country_norm": df.loc[mask, "country_norm"].to_numpy(copy=False),
        "key_value": values.loc[mask].to_numpy(copy=False),
    })


def _pairs_for_key(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                   key_type: str) -> pd.DataFrame:
    """Join one key family, dropping generic other-side postings immediately."""
    s1_keys = _blocking_key_frame(s1_df, key_type)
    other_keys = _blocking_key_frame(other_df, key_type)
    counts = other_keys.groupby(["country_norm", "key_value"], sort=False).size()
    keep = counts[counts <= MAX_BLOCK_SIZE].index
    if len(keep) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "entity_id"])

    keep_df = keep.to_frame(index=False)
    other_keys = other_keys.merge(keep_df, on=["country_norm", "key_value"], how="inner")
    pairs = s1_keys.merge(
        other_keys, on=["country_norm", "key_value"], suffixes=("_s1", "_other")
    )[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "entity_id"]
    return pairs


def _jaccard(a, b):
    if isinstance(a, str):
        a = a.split()
    if isinstance(b, str):
        b = b.split()
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _quick_scores(name_s1, name_other, addr_s1, addr_other, postal_s1, postal_other):
    """
    Cheap 0-1 similarity used only to rank/prune candidates, not to classify.
    Vectorized as plain Python zip loops (NOT DataFrame.apply(axis=1), which
    has heavy per-row overhead and is the #1 blocking-stage bottleneck at
    scale) — this is 5-10x faster for millions of candidate rows.
    """
    out = [0.0] * len(name_s1)
    for i, (ns1, no, as1, ao, ps1, po) in enumerate(
        zip(name_s1, name_other, addr_s1, addr_other, postal_s1, postal_other)
    ):
        name_j = _jaccard(ns1, no)
        addr_j = _jaccard(as1, ao)
        postal_match = 1.0 if ps1 and ps1 == po else 0.0
        out[i] = 0.55 * name_j + 0.25 * addr_j + 0.20 * postal_match
    return out


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                         top_k: int = TOP_K_CANDIDATES) -> pd.DataFrame:
    """
    s1_df, other_df: already `enrich()`-ed dataframes. other_df may contain
    both Source-2 and Source-3 records concatenated together (entity_id
    prefix disambiguates them downstream).

    Returns columns: source1_entity_id, entity_id (candidate), quick_score
    """
    _log(f"building blocking keys for {len(s1_df):,} S1 and {len(other_df):,} S2/S3 records...")
    pairs = None
    for key_type in ("name_first", "name_pair", "postal", "acronym"):
        current = _pairs_for_key(s1_df, other_df, key_type)
        _log(f"{key_type}: {len(current):,} candidate pairs")
        if pairs is None:
            pairs = current
        else:
            pairs = pd.concat([pairs, current], ignore_index=True).drop_duplicates(
                ignore_index=True
            )
            del current
    _log(f"{len(pairs):,} raw candidate pairs after union of all blocking keys")

    if pairs.empty:
        return pairs.assign(quick_score=[])

    s1_small = s1_df[["entity_id", "name_core_tokens", "addr_core_tokens", "addr_postal"]]
    other_small = other_df[["entity_id", "name_core_tokens", "addr_core_tokens", "addr_postal"]]

    merged = pairs.merge(
        s1_small, left_on="source1_entity_id", right_on="entity_id", suffixes=("", "_dropme")
    ).drop(columns=["entity_id_dropme"])
    merged = merged.rename(columns={
        "name_core_tokens": "name_core_tokens_s1",
        "addr_core_tokens": "addr_core_tokens_s1",
        "addr_postal": "addr_postal_s1",
    })
    merged = merged.merge(other_small, on="entity_id", suffixes=("", ""))
    merged = merged.rename(columns={
        "name_core_tokens": "name_core_tokens_other",
        "addr_core_tokens": "addr_core_tokens_other",
        "addr_postal": "addr_postal_other",
    })
    del pairs, s1_small, other_small

    _log("scoring candidate pairs for pruning...")
    merged["quick_score"] = _quick_scores(
        merged["name_core_tokens_s1"], merged["name_core_tokens_other"],
        merged["addr_core_tokens_s1"], merged["addr_core_tokens_other"],
        merged["addr_postal_s1"], merged["addr_postal_other"],
    )
    merged = merged.sort_values(["source1_entity_id", "quick_score"], ascending=[True, False])
    pruned = merged.groupby("source1_entity_id", sort=False).head(top_k)
    _log(f"pruned to {len(pruned):,} candidate pairs (top_k={top_k} per S1 entity)")
    result = pruned[["source1_entity_id", "entity_id", "quick_score"]].reset_index(drop=True)
    del merged, pruned
    return result
