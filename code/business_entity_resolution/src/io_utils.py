"""TSV I/O helpers. Output writing is done manually (not via csv.writer) so we
have exact control over the format the validator expects: tab between the two
columns, plain comma-joined IDs, no quoting."""
import os
import time
import pandas as pd


def _log(msg):
    print(f"[io] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def read_source(path: str) -> pd.DataFrame:
    _log(f"reading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    _log(f"  -> {len(df):,} rows")
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    return df


def write_id_list_file(path: str, header: tuple, mapping: dict, required_s1_ids):
    """mapping: {source1_entity_id: [id1, id2, ...]} (already deduped, ordered).
    required_s1_ids: iterable of every S1 id that must get a row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header[0] + "\t" + header[1] + "\n")
        for s1 in required_s1_ids:
            ids = mapping.get(s1, [])
            f.write(f"{s1}\t{','.join(ids)}\n")
