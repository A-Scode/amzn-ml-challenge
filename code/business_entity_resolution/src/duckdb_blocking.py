"""Disk-backed ingestion and candidate generation for large entity datasets."""
import json
import os
import shutil
import tempfile
import time

import duckdb
import pandas as pd

from blocking import MAX_BLOCK_SIZE, TOP_K_CANDIDATES, enrich


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
NORMALIZED_COLUMNS = [
    "entity_id", "name_clean", "name_core", "name_core_tokens",
    "name_first_token", "name_acronym", "addr_clean", "addr_postal",
    "addr_core_tokens", "country_norm",
]


def _log(message):
    print(f"[duckdb] {time.strftime('%H:%M:%S')}  {message}", flush=True)


def _cache_signature(source_paths, top_k):
    return json.dumps({
        "version": 4,
        "top_k": int(top_k),
        "sources": [
            {
                "path": os.path.abspath(path),
                "mtime_ns": os.stat(path).st_mtime_ns,
                "size": os.path.getsize(path),
            }
            for path in source_paths
        ],
    }, sort_keys=True)


def _cache_is_complete(connection, signature):
    tables = {
        row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    required_tables = {"source1", "other", "candidates", "pipeline_metadata"}
    if not required_tables.issubset(tables):
        return False
    row = connection.execute(
        "SELECT signature FROM pipeline_metadata LIMIT 1"
    ).fetchone()
    return row is not None and row[0] == signature


def _key_expression(key_type, alias):
    if key_type == "name_first":
        return f"{alias}.name_first_token", f"length({alias}.name_first_token) >= 3"
    if key_type == "name_pair":
        tokens = f"string_split({alias}.name_core_tokens, ' ')"
        first_three = f"list_slice({tokens}, 1, 3)"
        value = f"array_to_string(list_slice(list_sort({first_three}), 1, 2), '-')"
        return value, f"length({alias}.name_core_tokens) > 0 AND len({first_three}) >= 2"
    if key_type == "postal":
        return f"{alias}.addr_postal", f"length({alias}.addr_postal) >= 4"
    if key_type == "acronym":
        return f"{alias}.name_acronym", f"length({alias}.name_acronym) >= 2"
    raise ValueError(f"unknown blocking key: {key_type}")


def _ingest_source(connection, path, table_name, chunksize):
    connection.execute(
        f"CREATE TABLE {table_name} (" + ", ".join(
            f'"{column}" VARCHAR' for column in NORMALIZED_COLUMNS
        ) + ")"
    )
    rows_seen = 0
    for raw in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        usecols=SOURCE_COLUMNS,
        chunksize=chunksize,
    ):
        normalized = enrich(raw)
        connection.register("_normalized_batch", normalized)
        connection.execute(
            f"INSERT INTO {table_name} SELECT "
            + ", ".join(f'"{column}"' for column in NORMALIZED_COLUMNS)
            + " FROM _normalized_batch"
        )
        connection.unregister("_normalized_batch")
        rows_seen += len(normalized)
        _log(f"{table_name}: ingested {rows_seen:,} rows")
        del raw, normalized


def _token_list(alias, column):
    value = f"{alias}.{column}"
    return f"CASE WHEN {value} = '' THEN []::VARCHAR[] ELSE string_split({value}, ' ') END"


def _jaccard_sql(left_tokens, right_tokens):
    intersection = f"list_intersect({left_tokens}, {right_tokens})"
    union = f"list_distinct(list_concat({left_tokens}, {right_tokens}))"
    return (
        f"CASE WHEN len({union}) = 0 THEN 0.0 "
        f"ELSE len({intersection})::DOUBLE / len({union}) END"
    )


def _build_top_k(connection, top_k):
    """Rank bounded top-k lists per blocking family, then merge and rerank.

    A globally top-k candidate must be within its family's top-k; each pair
    can be proposed by at most the four distinct blocking families.
    """
    name_score = _jaccard_sql(
        _token_list("s", "name_core_tokens"),
        _token_list("o", "name_core_tokens"),
    )
    address_score = _jaccard_sql(
        _token_list("s", "addr_core_tokens"),
        _token_list("o", "addr_core_tokens"),
    )
    key_types = ("name_first", "name_pair", "postal", "acronym")
    proposal_queries = []
    for key_type in key_types:
        s1_value, s1_filter = _key_expression(key_type, "s")
        other_value, other_filter = _key_expression(key_type, "o")
        _log(f"precomputing valid {key_type} keys")
        connection.execute(f"""
            CREATE TABLE valid_{key_type} AS
            SELECT country_norm, {other_value} AS key_value
            FROM other o
            WHERE {other_filter}
            GROUP BY country_norm, key_value
            HAVING count(*) <= {MAX_BLOCK_SIZE}
        """)
        proposal_queries.append(f"""
            SELECT s.entity_id AS source1_entity_id, o.entity_id,
                '{key_type}' AS key_type
            FROM source1 s
            INNER JOIN valid_{key_type} v
                ON v.country_norm = s.country_norm AND v.key_value = {s1_value}
            INNER JOIN other o
                ON o.country_norm = v.country_norm AND {other_value} = v.key_value
            WHERE s.rowid >= {{batch_start}} AND s.rowid < {{batch_end}}
                AND {s1_filter} AND {other_filter}
        """)

    connection.execute(
        "CREATE TABLE candidates "
        "(source1_entity_id VARCHAR, entity_id VARCHAR, quick_score DOUBLE)"
    )
    s1_count = connection.execute("SELECT count(*) FROM source1").fetchone()[0]
    batch_size = max(1, int(os.environ.get("BER_BLOCK_S1_BATCH_SIZE", "10000")))
    _log(f"ranking candidates in S1 batches of {batch_size:,} ({s1_count:,} total)")

    for batch_start in range(0, s1_count, batch_size):
        batch_end = min(batch_start + batch_size, s1_count)
        proposals = " UNION ALL ".join(
            query.format(batch_start=batch_start, batch_end=batch_end)
            for query in proposal_queries
        )
        query = f"""
            INSERT INTO candidates
            WITH proposals AS ({proposals}), scored AS (
                SELECT p.source1_entity_id, p.entity_id, p.key_type,
                    0.55 * ({name_score})
                    + 0.25 * ({address_score})
                    + 0.20 * CASE
                        WHEN s.addr_postal <> '' AND s.addr_postal = o.addr_postal
                        THEN 1.0 ELSE 0.0 END AS quick_score
                FROM proposals p
                INNER JOIN source1 s ON s.entity_id = p.source1_entity_id
                INNER JOIN other o ON o.entity_id = p.entity_id
            ), top_per_key AS (
                SELECT source1_entity_id, key_type,
                    unnest(arg_min(
                        struct_pack(entity_id := entity_id, quick_score := quick_score),
                        struct_pack(rank_score := -quick_score, entity_id := entity_id),
                        {int(top_k)}
                    )) AS candidate
                FROM scored
                GROUP BY source1_entity_id, key_type
            ), unique_candidates AS (
                SELECT DISTINCT source1_entity_id,
                    candidate.entity_id AS entity_id,
                    candidate.quick_score AS quick_score
                FROM top_per_key
            ), ranked AS (
                SELECT source1_entity_id, entity_id, quick_score,
                    row_number() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY quick_score DESC, entity_id ASC
                    ) AS candidate_rank
                FROM unique_candidates
            )
            SELECT source1_entity_id, entity_id, quick_score
            FROM ranked
            WHERE candidate_rank <= {int(top_k)}
        """
        connection.execute(query)
        _log(f"ranked S1 rows {batch_end:,}/{s1_count:,}")

    for key_type in key_types:
        connection.execute(f"DROP TABLE valid_{key_type}")


def open_blocking_database(data_dir, split, top_k=TOP_K_CANDIDATES, cache_dir=None):
    """Build/reuse a DuckDB candidate table and return its connection and IDs.

    The caller owns the returned connection and must close it. When caching is
    disabled, the returned temporary directory must also be removed by caller.
    """
    source_paths = [
        os.path.join(data_dir, f"{split}_source{index}.tsv")
        for index in (1, 2, 3)
    ]
    owned_temp_dir = None
    if cache_dir:
        work_dir = cache_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        owned_temp_dir = tempfile.mkdtemp(prefix="ber-duckdb-")
        work_dir = owned_temp_dir

    default_spill_dir = os.path.join(os.path.dirname(__file__), "duckdb_temp")
    spill_dir = os.environ.get("BER_DUCKDB_TEMP_DIR", default_spill_dir)
    os.makedirs(spill_dir, exist_ok=True)
    if not os.access(spill_dir, os.W_OK):
        raise OSError(f"DuckDB spill directory is not writable: {spill_dir}")

    db_path = os.path.join(work_dir, f"{split}_duckdb_v4_top{top_k}.db")
    signature = _cache_signature(source_paths, top_k)
    is_fresh = os.path.exists(db_path) and all(
        os.path.getmtime(db_path) >= os.path.getmtime(path) for path in source_paths
    )

    if is_fresh:
        try:
            cached_connection = duckdb.connect(db_path, read_only=True)
            is_fresh = _cache_is_complete(cached_connection, signature)
            cached_connection.close()
        except duckdb.Error:
            is_fresh = False

    if not is_fresh:
        for suffix in ("", ".wal"):
            path = db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    memory_limit = os.environ.get("BER_DUCKDB_MEMORY_LIMIT", "8GB")
    threads = max(1, int(os.environ.get("BER_DUCKDB_THREADS", "2")))
    connection = duckdb.connect(
        db_path,
        config={
            "memory_limit": memory_limit,
            "threads": str(threads),
            "temp_directory": spill_dir,
        },
    )
    try:
        if not is_fresh:
            _log(f"building normalized tables; memory_limit={memory_limit}, threads={threads}")
            _ingest_source(connection, source_paths[0], "source1", chunksize=20_000)
            connection.execute("CREATE TABLE other AS SELECT * FROM source1 WHERE false")
            for source_path in source_paths[1:]:
                temporary_table = "_source_batch"
                connection.execute(
                    f"CREATE TABLE {temporary_table} (" + ", ".join(
                        f'"{column}" VARCHAR' for column in NORMALIZED_COLUMNS
                    ) + ")"
                )
                rows_seen = 0
                for raw in pd.read_csv(
                    source_path,
                    sep="\t",
                    dtype=str,
                    keep_default_na=False,
                    na_values=[],
                    usecols=SOURCE_COLUMNS,
                    chunksize=20_000,
                ):
                    normalized = enrich(raw)
                    connection.register("_normalized_batch", normalized)
                    connection.execute(
                        f"INSERT INTO {temporary_table} SELECT "
                        + ", ".join(f'"{column}"' for column in NORMALIZED_COLUMNS)
                        + " FROM _normalized_batch"
                    )
                    connection.unregister("_normalized_batch")
                    rows_seen += len(normalized)
                    _log(f"{os.path.basename(source_path)}: ingested {rows_seen:,} rows")
                    del raw, normalized
                connection.execute(f"INSERT INTO other SELECT * FROM {temporary_table}")
                connection.execute(f"DROP TABLE {temporary_table}")

            _build_top_k(connection, top_k)
            connection.execute(
                "CREATE TABLE pipeline_metadata (signature VARCHAR NOT NULL)"
            )
            connection.execute(
                "INSERT INTO pipeline_metadata VALUES (?)", [signature]
            )
            connection.execute("CHECKPOINT")
            is_fresh = True

        candidates = connection.execute(
            "SELECT source1_entity_id, entity_id, quick_score "
            "FROM candidates ORDER BY source1_entity_id, quick_score DESC, entity_id"
        ).fetch_df()
        source1_ids = connection.execute(
            "SELECT entity_id FROM source1 ORDER BY rowid"
        ).fetchall()
        source1_ids = [row[0] for row in source1_ids]
        _log(f"candidate table ready: {len(candidates):,} pairs")
        return connection, candidates, source1_ids, owned_temp_dir
    except BaseException:
        connection.close()
        if not is_fresh:
            for suffix in ("", ".wal"):
                path = db_path + suffix
                if os.path.exists(path):
                    os.remove(path)
        if owned_temp_dir:
            shutil.rmtree(owned_temp_dir, ignore_errors=True)
        raise
