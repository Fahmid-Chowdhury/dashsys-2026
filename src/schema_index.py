import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb


ID_COLUMN_PATTERN = re.compile(r"(^id$|id$|_id$)", re.IGNORECASE)


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def register_parquet_views(
    con: duckdb.DuckDBPyConnection,
    snapshot_dir: str | Path
) -> tuple[List[str], List[Dict[str, str]]]:
    snapshot_dir = Path(snapshot_dir)

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"DBSnapshot folder not found: {snapshot_dir}")

    table_names: List[str] = []
    skipped_files: List[Dict[str, str]] = []

    for fname in os.listdir(snapshot_dir):
        if not fname.endswith(".parquet"):
            continue

        table_name = fname[: -len(".parquet")]
        parquet_path = snapshot_dir / fname

        try:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM read_parquet('{parquet_path.as_posix()}')
                """
            )

            table_names.append(table_name)

        except Exception as e:
            skipped_files.append(
                {
                    "file": fname,
                    "reason": str(e).splitlines()[0]
                }
            )
            print(f"[WARN] Skipped {fname}: {str(e).splitlines()[0]}")

    if not table_names:
        raise ValueError(f"No valid parquet files found in: {snapshot_dir}")

    return sorted(table_names), skipped_files


def get_table_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str
) -> List[Dict[str, Any]]:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()

    columns = []

    for row in rows:
        # DuckDB PRAGMA table_info returns:
        # cid, name, type, notnull, dflt_value, pk
        columns.append(
            {
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "primary_key": bool(row[5]),
            }
        )

    return columns


def get_row_count(
    con: duckdb.DuckDBPyConnection,
    table_name: str
) -> int:
    result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(result[0]) if result else 0


def detect_table_kind(table_name: str) -> str:
    lower = table_name.lower()

    if lower.startswith("dim_"):
        return "dimension"

    if lower.startswith("hkg_br_") or lower.startswith("br_"):
        return "bridge"

    return "other"


def detect_domain(table_name: str) -> str:
    name = table_name.lower()

    domain_rules = [
        ("journey", ["campaign", "journey"]),
        ("segment", ["segment", "audience"]),
        ("dataset", ["collection", "dataset"]),
        ("schema", ["blueprint", "schema"]),
        ("connector", ["connector", "source"]),
        ("destination", ["target", "destination"]),
        ("property", ["property", "field", "attribute"]),
        ("dataflow", ["dataflow", "flow"]),
    ]

    for domain, keywords in domain_rules:
        if any(keyword in name for keyword in keywords):
            return domain

    return "general"


def detect_id_columns(columns: List[Dict[str, Any]]) -> List[str]:
    id_columns = []

    for col in columns:
        col_name = col["name"]
        if ID_COLUMN_PATTERN.search(col_name):
            id_columns.append(col_name)

    return sorted(set(id_columns))


def get_sample_values(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    limit: int = 5
) -> List[Any]:
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            LIMIT {limit}
            """
        ).fetchall()

        return [row[0] for row in rows]

    except Exception:
        return []


def build_table_index(
    con: duckdb.DuckDBPyConnection,
    table_names: List[str],
    include_samples: bool = True
) -> Dict[str, Any]:
    table_index: Dict[str, Any] = {}

    for table_name in table_names:
        columns = get_table_columns(con, table_name)
        row_count = get_row_count(con, table_name)
        table_kind = detect_table_kind(table_name)
        domain = detect_domain(table_name)
        id_columns = detect_id_columns(columns)

        column_records = []

        for col in columns:
            column_record = {
                "name": col["name"],
                "type": col["type"],
                "not_null": col["not_null"],
                "primary_key": col["primary_key"],
            }

            if include_samples:
                column_record["sample_values"] = get_sample_values(
                    con,
                    table_name,
                    col["name"],
                    limit=3
                )

            column_records.append(column_record)

        table_index[table_name] = {
            "table_name": table_name,
            "kind": table_kind,
            "domain": domain,
            "row_count": row_count,
            "columns": column_records,
            "id_columns": id_columns,
        }

    return table_index


def collect_column_map(
    table_index: Dict[str, Any]
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Builds:
    {
      "segmentid": [
        ("dim_segment", "segmentid"),
        ("hkg_br_segment_target", "segmentid")
      ]
    }
    """

    column_map: Dict[str, List[Tuple[str, str]]] = {}

    for table_name, table_info in table_index.items():
        for col in table_info["columns"]:
            col_name = col["name"]
            normalized = col_name.lower()

            if normalized not in column_map:
                column_map[normalized] = []

            column_map[normalized].append((table_name, col_name))

    return column_map


def build_join_candidates(table_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    column_map = collect_column_map(table_index)
    joins: List[Dict[str, Any]] = []

    for normalized_col, locations in column_map.items():
        if len(locations) < 2:
            continue

        if not ID_COLUMN_PATTERN.search(normalized_col):
            continue

        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                left_table, left_col = locations[i]
                right_table, right_col = locations[j]

                left_kind = table_index[left_table]["kind"]
                right_kind = table_index[right_table]["kind"]

                confidence = "medium"
                reason = "same ID-like column name"

                if {left_kind, right_kind} == {"dimension", "bridge"}:
                    confidence = "high"
                    reason = "dimension-to-bridge join using same ID column"

                elif left_kind == "bridge" and right_kind == "bridge":
                    confidence = "medium"
                    reason = "bridge-to-bridge join using same ID column"

                joins.append(
                    {
                        "left_table": left_table,
                        "left_column": left_col,
                        "right_table": right_table,
                        "right_column": right_col,
                        "confidence": confidence,
                        "reason": reason,
                    }
                )

    return joins


def build_schema_index(
    snapshot_dir: str | Path,
    include_samples: bool = True
) -> Dict[str, Any]:
    con = connect_duckdb()

    try:
        table_names, skipped_files = register_parquet_views(con, snapshot_dir)
        table_index = build_table_index(
            con=con,
            table_names=table_names,
            include_samples=include_samples,
        )
        join_candidates = build_join_candidates(table_index)

        return {
            "snapshot_dir": str(snapshot_dir),
            "table_count": len(table_names),
            "skipped_files": skipped_files,
            "tables": table_index,
            "join_candidates": join_candidates,
        }

    finally:
        con.close()


def save_schema_index(
    schema_index: Dict[str, Any],
    output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schema_index, f, indent=2, ensure_ascii=False, default=str)


def print_schema_summary(schema_index: Dict[str, Any]) -> None:
    print("Schema index summary")
    print("====================")
    print(f"Snapshot directory: {schema_index['snapshot_dir']}")
    print(f"Table count: {schema_index['table_count']}")
    
    skipped_files = schema_index.get("skipped_files", [])
    print(f"Skipped parquet files: {len(skipped_files)}")

    if skipped_files:
        print("\nSkipped files:")
        for item in skipped_files:
            print(f"  - {item['file']}: {item['reason']}")
    
    print(f"Join candidates: {len(schema_index['join_candidates'])}")

    print("\nTables:")
    for table_name, info in schema_index["tables"].items():
        print(
            f"  - {table_name} "
            f"({info['kind']}, domain={info['domain']}, "
            f"rows={info['row_count']}, columns={len(info['columns'])})"
        )

    print("\nHigh-confidence joins:")
    high_joins = [
        join for join in schema_index["join_candidates"]
        if join["confidence"] == "high"
    ]

    for join in high_joins[:20]:
        print(
            f"  - {join['left_table']}.{join['left_column']} "
            f"= {join['right_table']}.{join['right_column']}"
        )

    if len(high_joins) > 20:
        print(f"  ... and {len(high_joins) - 20} more")


if __name__ == "__main__":
    schema_index = build_schema_index(
        snapshot_dir="DBSnapshot",
        include_samples=True,
    )

    save_schema_index(schema_index, "data/schema_index.json")
    print_schema_summary(schema_index)

    print("\nSaved to: data/schema_index.json")