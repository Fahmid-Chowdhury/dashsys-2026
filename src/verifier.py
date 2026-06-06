import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


FORBIDDEN_SQL_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "copy", "export", "import", "pragma", "call"
}

SQL_TABLE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

TABLE_ALIAS_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

QUALIFIED_COLUMN_PATTERN = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"
)

PLACEHOLDER_PATTERN = re.compile(r"\{[^{}]+\}|<[^<>]+>")


@dataclass
class VerificationResult:
    ok: bool
    target: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repair_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "target": self.target,
            "errors": self.errors,
            "warnings": self.warnings,
            "repair_hint": self.repair_hint,
        }


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    return name.strip().strip('"').strip("'").lower()


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def has_multiple_sql_statements(sql: str) -> bool:
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]

    return ";" in stripped


def is_readonly_sql(sql: str) -> bool:
    sql_clean = normalize_sql(sql).lower()

    if not (sql_clean.startswith("select") or sql_clean.startswith("with")):
        return False

    tokens = set(re.findall(r"\b[a-zA-Z_]+\b", sql_clean))

    return not bool(tokens & FORBIDDEN_SQL_KEYWORDS)


def extract_sql_tables(sql: str) -> List[str]:
    return sorted(set(normalize_name(t) for t in SQL_TABLE_PATTERN.findall(sql)))


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """
    Returns alias -> table.

    Example:
    FROM DIM_CAMPAIGN AS CAMPAIGN
    gives:
    {"campaign": "dim_campaign"}
    """

    aliases: Dict[str, str] = {}

    for table, alias in TABLE_ALIAS_PATTERN.findall(sql):
        table_norm = normalize_name(table)
        alias_norm = normalize_name(alias)

        # Avoid treating SQL keywords as aliases.
        if alias_norm in {"where", "join", "on", "limit", "order", "group"}:
            continue

        aliases[alias_norm] = table_norm

    return aliases


def get_allowed_table_column_map(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    allowed_tables = metadata.get("allowed_tables", {}) or {}
    result: Dict[str, List[str]] = {}

    for table_name, table_info in allowed_tables.items():
        columns = table_info.get("columns", []) or []

        result[normalize_name(table_name)] = [
            normalize_name(col.get("name", ""))
            for col in columns
            if col.get("name")
        ]

    return result


def format_allowed_columns_for_feedback(
    table_column_map: Dict[str, List[str]],
    max_columns_per_table: int = 30,
) -> str:
    lines = []

    for table, columns in table_column_map.items():
        shown = columns[:max_columns_per_table]
        lines.append(f"- {table}: {shown}")

    return "\n".join(lines)


def validate_qualified_columns(
    sql: str,
    table_column_map: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    aliases = extract_table_aliases(sql)
    tables_in_sql = extract_sql_tables(sql)

    # Table names can refer to themselves as qualifiers too.
    table_or_alias_to_table = {table: table for table in tables_in_sql}
    table_or_alias_to_table.update(aliases)

    for qualifier, column in QUALIFIED_COLUMN_PATTERN.findall(sql):
        qualifier_norm = normalize_name(qualifier)
        column_norm = normalize_name(column)

        # Skip likely function/schema names if not table aliases.
        if qualifier_norm not in table_or_alias_to_table:
            warnings.append(
                f"Qualifier '{qualifier}' is not a known table alias from this SQL."
            )
            continue

        table_name = table_or_alias_to_table[qualifier_norm]
        allowed_columns = table_column_map.get(table_name, [])

        if allowed_columns and column_norm not in allowed_columns:
            errors.append(
                f"Column '{column}' does not exist in allowed table '{table_name}'."
            )

    return errors, warnings


def register_snapshot_for_dry_run(snapshot_dir: str | Path):
    import duckdb

    snapshot_dir = Path(snapshot_dir)
    con = duckdb.connect(database=":memory:")

    if not snapshot_dir.exists():
        return con, [f"Snapshot directory not found: {snapshot_dir}. DuckDB dry-run skipped."]

    warnings = []

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
        except Exception as e:
            warnings.append(f"Skipped {fname} during dry-run registration: {str(e).splitlines()[0]}")

    return con, warnings


def duckdb_dry_run_sql(
    sql: str,
    snapshot_dir: str | Path = "DBSnapshot",
) -> tuple[bool, List[str], List[str]]:
    """
    Uses DuckDB EXPLAIN to validate syntax, table names, and column names
    without executing the query.
    """

    try:
        con, warnings = register_snapshot_for_dry_run(snapshot_dir)

        try:
            con.execute(f"EXPLAIN {sql}")
            return True, [], warnings

        except Exception as e:
            return False, [f"DuckDB dry-run failed: {str(e).splitlines()[0]}"], warnings

        finally:
            con.close()

    except ImportError:
        return True, [], ["DuckDB is not installed. Dry-run validation skipped."]

    except Exception as e:
        return True, [], [f"DuckDB dry-run skipped due to setup error: {str(e).splitlines()[0]}"]


def validate_sql_call(
    sql: str,
    metadata: Dict[str, Any],
    snapshot_dir: str | Path = "DBSnapshot",
    enable_duckdb_dry_run: bool = True,
) -> VerificationResult:
    result = VerificationResult(ok=True, target="sql")

    sql = sql.strip()

    if not sql:
        return VerificationResult(
            ok=False,
            target="sql",
            errors=["SQL query is empty."],
            repair_hint="Generate one read-only SELECT query using only allowed tables.",
        )

    budget = metadata.get("tool_budget", {}) or {}
    if budget.get("max_sql_calls", 0) <= 0:
        return VerificationResult(
            ok=False,
            target="sql",
            errors=["The selected route does not allow SQL calls."],
            repair_hint="Do not call execute_sql for this query. Use the allowed API route.",
        )

    if has_multiple_sql_statements(sql):
        result.errors.append("SQL contains multiple statements.")

    if not is_readonly_sql(sql):
        result.errors.append(
            "SQL must be read-only and start with SELECT or WITH. Unsafe SQL keyword detected."
        )

    tables_in_sql = extract_sql_tables(sql)
    if not tables_in_sql:
        result.errors.append("No table found in SQL. Use FROM with an allowed table.")

    table_column_map = get_allowed_table_column_map(metadata)
    allowed_tables = set(table_column_map.keys())

    for table in tables_in_sql:
        if table not in allowed_tables:
            result.errors.append(
                f"Table '{table}' is not allowed for this query. "
                f"Allowed tables: {sorted(allowed_tables)}"
            )

    column_errors, column_warnings = validate_qualified_columns(sql, table_column_map)
    result.errors.extend(column_errors)
    result.warnings.extend(column_warnings)

    if not result.errors and enable_duckdb_dry_run:
        dry_ok, dry_errors, dry_warnings = duckdb_dry_run_sql(
            sql=sql,
            snapshot_dir=snapshot_dir,
        )

        result.warnings.extend(dry_warnings)

        if not dry_ok:
            result.errors.extend(dry_errors)

    if result.errors:
        result.ok = False
        allowed_column_hint = format_allowed_columns_for_feedback(table_column_map)

        result.repair_hint = (
            "Rewrite the SQL using only allowed tables and columns from metadata. "
            "Do not use user-facing field names unless they exactly exist as SQL columns. "
            "Map requested fields to the closest actual columns. "
            "Make sure the SQL parses in DuckDB and does not invent tables, columns, joins, or filters.\n\n"
            f"Allowed columns by table:\n{allowed_column_hint}"
        )

    return result


def normalize_api_path(url_or_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Supports both:
    /ajo/journey?pageSize=10
    https://platform.adobe.io/ajo/journey?pageSize=10
    """

    parsed = urlparse(url_or_path)

    if parsed.scheme and parsed.netloc:
        path = parsed.path
        query_string = parsed.query
    else:
        if "?" in url_or_path:
            path, query_string = url_or_path.split("?", 1)
        else:
            path, query_string = url_or_path, ""

    if not path.startswith("/"):
        path = "/" + path

    path = path.rstrip("/") if path != "/" else path

    raw_params = parse_qs(query_string, keep_blank_values=True)
    params = {
        key: values[0] if values else ""
        for key, values in raw_params.items()
    }

    return path, params


def endpoint_template_to_regex(template_path: str) -> re.Pattern:
    escaped = re.escape(template_path)

    # Convert \{schema_id\} or \{SCHEMA_ID\} to one path segment.
    escaped = re.sub(r"\\\{[^{}]+\\\}", r"[^/]+", escaped)

    return re.compile("^" + escaped + "$")


def path_matches_template(path: str, template_path: str) -> bool:
    if path == template_path:
        return True

    pattern = endpoint_template_to_regex(template_path)
    return bool(pattern.match(path))


def has_unfilled_placeholder(path: str) -> bool:
    return bool(PLACEHOLDER_PATTERN.search(path))


def enrich_allowed_endpoint_records(
    metadata: Dict[str, Any],
    api_index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    records = metadata.get("allowed_api_endpoints", []) or []

    if not api_index:
        return records

    endpoint_index = api_index.get("endpoints", {}) or {}
    enriched = []

    for record in records:
        endpoint = record.get("endpoint")

        if endpoint in endpoint_index:
            merged = {**record, **endpoint_index[endpoint]}
            enriched.append(merged)
        else:
            enriched.append(record)

    return enriched


def get_query_param_names(endpoint_record: Dict[str, Any]) -> List[str]:
    openapi = endpoint_record.get("openapi")

    if openapi:
        names = openapi.get("query_param_names", []) or []
        if names:
            return names

    return endpoint_record.get("param_names", []) or []


def get_required_query_param_names(endpoint_record: Dict[str, Any]) -> List[str]:
    openapi = endpoint_record.get("openapi")

    if not openapi:
        return []

    query_params = openapi.get("query_params", []) or []

    return sorted(
        p.get("name")
        for p in query_params
        if p.get("name") and p.get("required")
    )


def get_path_param_names(endpoint_record: Dict[str, Any]) -> List[str]:
    openapi = endpoint_record.get("openapi")

    if openapi:
        names = openapi.get("path_param_names", []) or []
        if names:
            return names

    path = endpoint_record.get("path", "")
    return re.findall(r"\{([^{}]+)\}", path)


def find_matching_endpoint(
    method: str,
    path: str,
    allowed_records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    method = method.upper()

    for record in allowed_records:
        if record.get("method", "").upper() != method:
            continue

        template_path = record.get("path", "")

        if path_matches_template(path, template_path):
            return record

    return None


def find_placeholders_in_params(params: Dict[str, Any]) -> List[str]:
    errors = []

    for key, value in params.items():
        key_text = str(key)
        value_text = str(value)

        if PLACEHOLDER_PATTERN.search(key_text):
            errors.append(f"API parameter name contains unfilled placeholder: {key_text}")

        if PLACEHOLDER_PATTERN.search(value_text):
            errors.append(f"API parameter '{key_text}' contains unfilled placeholder: {value_text}")

    return errors


def validate_api_call(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    api_index: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    result = VerificationResult(ok=True, target="api")

    method = method.upper().strip()
    path, url_params = normalize_api_path(url)

    merged_params: Dict[str, Any] = {}
    merged_params.update(url_params)

    if params:
        merged_params.update(params)
        
    param_placeholder_errors = find_placeholders_in_params(merged_params)

    if param_placeholder_errors:
        return VerificationResult(
            ok=False,
            target="api",
            errors=param_placeholder_errors,
            repair_hint="Fill all placeholder values in API query parameters before calling the API.",
        )

    budget = metadata.get("tool_budget", {}) or {}

    if budget.get("max_api_calls", 0) <= 0:
        return VerificationResult(
            ok=False,
            target="api",
            errors=["The selected route does not allow API calls."],
            repair_hint="Do not call call_api for this query. Use the allowed SQL route.",
        )

    if has_unfilled_placeholder(path):
        return VerificationResult(
            ok=False,
            target="api",
            errors=[f"API path contains unfilled placeholder: {path}"],
            repair_hint="Fill all path parameters before calling the API.",
        )

    allowed_records = enrich_allowed_endpoint_records(metadata, api_index)

    if not allowed_records:
        return VerificationResult(
            ok=False,
            target="api",
            errors=["No API endpoint is allowed by metadata."],
            repair_hint="Use only endpoints listed under allowed_api_endpoints.",
        )

    matched = find_matching_endpoint(method, path, allowed_records)

    if not matched:
        allowed = [
            record.get("endpoint", "")
            for record in allowed_records
        ]

        return VerificationResult(
            ok=False,
            target="api",
            errors=[
                f"API endpoint '{method} {path}' is not allowed for this query.",
                f"Allowed endpoints: {allowed}",
            ],
            repair_hint="Choose one of the allowed API endpoints from metadata.",
        )

    allowed_query_params = set(get_query_param_names(matched))
    required_query_params = set(get_required_query_param_names(matched))
    path_param_names = get_path_param_names(matched)

    # If OpenAPI has path params, the matched concrete path is enough.
    # We still warn if template exists but concrete path did not replace it.
    if path_param_names and has_unfilled_placeholder(path):
        result.errors.append(
            f"Unfilled path params remain in path. Required path params: {path_param_names}"
        )

    if allowed_query_params:
        for key in merged_params:
            if key not in allowed_query_params:
                result.errors.append(
                    f"Query parameter '{key}' is not allowed for endpoint {matched.get('endpoint')}."
                )

    missing_required = [
        key for key in required_query_params
        if key not in merged_params
    ]

    if missing_required:
        result.errors.append(
            f"Missing required query parameters: {missing_required}"
        )

    if result.errors:
        result.ok = False
        result.repair_hint = (
            "Rewrite the API call using only the matched endpoint and allowed parameters. "
            "Do not leave placeholders unfilled."
        )

    return result


def validate_tool_call(
    tool_call: Dict[str, Any],
    metadata: Dict[str, Any],
    api_index: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    """
    Expected SQL call:
    {
      "action": "sql_query",
      "sql": "SELECT ..."
    }

    Expected API call:
    {
      "action": "api_call",
      "method": "GET",
      "url": "/ajo/journey",
      "params": {"pageSize": "10"}
    }
    """

    action = tool_call.get("action")

    if action == "sql_query":
        return validate_sql_call(
            sql=tool_call.get("sql", ""),
            metadata=metadata,
        )

    if action == "api_call":
        return validate_api_call(
            method=tool_call.get("method", "GET"),
            url=tool_call.get("url", ""),
            params=tool_call.get("params") or {},
            metadata=metadata,
            api_index=api_index,
        )

    return VerificationResult(
        ok=False,
        target="unknown",
        errors=[f"Unknown tool action: {action}"],
        repair_hint="Use either sql_query or api_call.",
    )


def load_api_index_if_available(
    path: str | Path = "data/api_index_enriched.json",
) -> Optional[Dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return None

    return load_json(path)