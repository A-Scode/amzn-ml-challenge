"""
Lightweight disk cache for the pipeline's expensive stages (normalization,
blocking, feature computation). Each cache entry is a pickled DataFrame keyed
by a path; it's considered fresh as long as it's newer than every one of the
given source files, so editing/replacing the raw .tsv files automatically
invalidates it — no manual cache-busting needed.
"""
import os
import pickle
import time

import pandas as pd


def _log(msg):
    print(f"[cache] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _is_fresh(cache_path: str, source_paths) -> bool:
    if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for p in source_paths:
        if os.path.exists(p) and os.path.getmtime(p) > cache_mtime:
            return False
    return True


def load_or_build(cache_path: str, source_paths, build_fn, label: str = ""):
    """
    Return (df, was_recomputed). Reuses `cache_path` if it's newer than every
    path in `source_paths`; otherwise calls `build_fn()`, saves the result,
    and returns it.
    """
    if cache_path is None:
        return build_fn(), True

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if _is_fresh(cache_path, source_paths):
        try:
            _log(f"reusing cached {label} -> {cache_path}")
            return pd.read_pickle(cache_path), False
        except (EOFError, pickle.UnpicklingError, ValueError) as exc:
            _log(f"discarding invalid cache for {label}: {exc}")

    _log(f"no fresh cache for {label}, computing...")
    df = build_fn()
    # Write beside the destination and replace it only after serialization
    # succeeds, so an interrupted run cannot leave a readable-looking partial
    # cache file behind.
    temp_path = f"{cache_path}.tmp-{os.getpid()}"
    try:
        _log(f"writing cache {label} -> {temp_path}")
        df.to_pickle(temp_path)
        os.replace(temp_path, cache_path)
    except KeyboardInterrupt:
        _log(f"cache write interrupted for {label}; partial file removed")
        raise
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    _log(f"cached {label} -> {cache_path}  ({os.path.getsize(cache_path) / 1e6:.1f} MB)")
    return df, True
