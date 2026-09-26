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
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from normalize import normalize_name, normalize_address


def _log(msg):
    print(f"[blocking] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _normalize_name_chunk(chunk):
    return [normalize_name(x) for x in chunk]


def _normalize_addr_chunk(chunk):
    return [normalize_address(x) for x in chunk]


def _chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _parallel_map(func_chunk, values, chunk_size=200_000, n_workers=None, label=""):
    """Split `values` into chunks and normalize them across processes.

    Falls back to a plain single-process loop (still chunked, so progress
    logging still works) if the dataset is small or multiprocessing can't be
    used in this environment.
    """
    # Forked workers duplicate pandas/string memory and can push a modest
    # machine into swap. Opt in to parallelism only when the host has room.
    n_workers = n_workers or int(os.environ.get("BER_NORMALIZE_WORKERS", "1"))
    n_workers = max(1, n_workers)
    _log(f"  {label}: n_workers={n_workers}")
    chunk_list = list(_chunks(values, chunk_size))
    n_chunks = len(chunk_list)
    if n_chunks <= 1 or n_workers <= 1:
        out = []
        for i, c in enumerate(chunk_list, 1):
            out.extend(func_chunk(c))
            _log(f"  {label}: chunk {i}/{n_chunks} done (single-process)")
        return out

    results = [None] * n_chunks
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(func_chunk, c): i for i, c in enumerate(chunk_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            _log(f"  {label}: chunk {done}/{n_chunks} done ({n_workers} workers)")
    out = []
    for r in results:
        out.extend(r)
    return out

MAX_BLOCK_SIZE = 800   # drop a key value if it matches more than this many S2/S3 records
TOP_K_CANDIDATES = 25  # final candidates kept per S1 entity


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized name/address fields to a source dataframe."""
    _log(f"normalizing {len(df):,} records (memory-safe chunked mode)...")
    # Keep only columns consumed by blocking and feature generation.
    df = df[["entity_id", "business_name", "business_address", "country"]].copy()
    names = _parallel_map(_normalize_name_chunk, df["business_name"].tolist(), label="names")
    addrs = _parallel_map(_normalize_addr_chunk, df["business_address"].tolist(), label="addresses")
    df["name_clean"] = [n["clean"] for n in names]
    df["name_core"] = [n["core"] for n in names]
    df["name_core_tokens"] = [n["core_tokens"] for n in names]
    df["name_first_token"] = [n["first_token"] for n in names]
    df["name_acronym"] = [n["acronym"] for n in names]
    df["addr_clean"] = [a["clean"] for a in addrs]
    df["addr_postal"] = [a["postal_code"] for a in addrs]
    df["addr_core_tokens"] = [a["core_tokens"] for a in addrs]
    df["country_norm"] = df["country"].astype(str).str.strip().str.lower()
    del names, addrs
    return df


def _first_two_sorted(tokens):
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
    key_pairs = []
    for key_type in ("name_first", "name_pair", "postal", "acronym"):
        current = _pairs_for_key(s1_df, other_df, key_type)
        _log(f"{key_type}: {len(current):,} candidate pairs")
        key_pairs.append(current)
    pairs = pd.concat(key_pairs, ignore_index=True).drop_duplicates()
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

    _log("scoring candidate pairs for pruning...")
    merged["quick_score"] = _quick_scores(
        merged["name_core_tokens_s1"], merged["name_core_tokens_other"],
        merged["addr_core_tokens_s1"], merged["addr_core_tokens_other"],
        merged["addr_postal_s1"], merged["addr_postal_other"],
    )
    merged = merged.sort_values(["source1_entity_id", "quick_score"], ascending=[True, False])
    pruned = merged.groupby("source1_entity_id", sort=False).head(top_k)
    _log(f"pruned to {len(pruned):,} candidate pairs (top_k={top_k} per S1 entity)")
    return pruned[["source1_entity_id", "entity_id", "quick_score"]].reset_index(drop=True)
