import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlparse

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import ExampleRecord, IndexedExample, ParsedApiCall, ToolStep


SQL_TABLE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


def load_examples(path: str | Path) -> List[ExampleRecord]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"data.json not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_examples = json.load(f)

    examples: List[ExampleRecord] = []

    for item in raw_examples:
        trace_steps: List[ToolStep] = []

        for step in item.get("trace", []):
            action = step.get("action", "")

            if action == "sql_query":
                trace_steps.append(
                    ToolStep(
                        step=int(step.get("step", 0)),
                        action=action,
                        sql=step.get("sql"),
                        status=step.get("status"),
                    )
                )

            elif action == "api_call":
                api_call = step.get("api_call", {}) or {}

                trace_steps.append(
                    ToolStep(
                        step=int(step.get("step", 0)),
                        action=action,
                        api_method=api_call.get("method"),
                        api_url=api_call.get("url"),
                        api_params=api_call.get("params") or {},
                        status=str(api_call.get("status_code")),
                    )
                )

        examples.append(
            ExampleRecord(
                query=item.get("query", "").strip(),
                answer=item.get("answer", "").strip(),
                gold_sql=item.get("gold_sql", "") or "",
                gold_api=item.get("gold_api", []) or [],
                trace=trace_steps,
            )
        )

    return examples


def extract_sql_tables(sql: str) -> List[str]:
    if not sql:
        return []

    tables = SQL_TABLE_PATTERN.findall(sql)
    return sorted(set(table.lower() for table in tables))


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        return None


def parse_gold_api(api_call: str) -> ParsedApiCall:
    """
    Handles examples like:

    GET /data/foundation/catalog/batches?limit=10&status=success

    POST /data/infrastructure/observability/insights/metrics body={...}
    """
    raw = api_call.strip()

    if not raw:
        return ParsedApiCall(
            raw="",
            method="UNKNOWN",
            path="",
            endpoint="UNKNOWN",
            params={},
            body=None,
        )

    parts = raw.split(maxsplit=1)

    if len(parts) == 1:
        return ParsedApiCall(
            raw=raw,
            method="UNKNOWN",
            path=raw,
            endpoint=f"UNKNOWN {raw}",
            params={},
            body=None,
        )

    method, rest = parts
    method = method.upper()

    body = None

    if " body=" in rest:
        url_part, body_text = rest.split(" body=", 1)
        body = safe_json_loads(body_text.strip())
    else:
        url_part = rest

    parsed_url = urlparse(url_part)

    path = parsed_url.path
    query_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))

    endpoint = f"{method} {path}"

    return ParsedApiCall(
        raw=raw,
        method=method,
        path=path,
        endpoint=endpoint,
        params=query_params,
        body=body,
    )


def infer_route(example: ExampleRecord) -> str:
    has_sql = bool(example.gold_sql.strip())
    api_count = len(example.gold_api)

    if has_sql and api_count == 0:
        return "SQL_ONLY"

    if not has_sql and api_count == 1:
        return "API_ONLY"

    if has_sql and api_count == 1:
        return "SQL_PLUS_API"

    if not has_sql and api_count > 1:
        return "API_CHAIN"

    if has_sql and api_count > 1:
        return "SQL_PLUS_API_CHAIN"

    return "UNKNOWN"


def infer_intent(query: str) -> str:
    q = query.lower()

    count_patterns = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\btotal number\b",
    ]

    status_patterns = [
        r"\bstatus\b",
        r"\bfailed\b",
        r"\bsuccess\b",
        r"\bsuccessful\b",
        r"\bprocessing\b",
        r"\bqueued\b",
        r"\binactive\b",
        r"\bactive\b",
        r"\bcompleted\b",
    ]

    field_patterns = [
        r"\bfield\b",
        r"\bproperty\b",
        r"\battribute\b",
        r"\bxdm path\b",
    ]

    time_patterns = [
        r"\bwhen\b",
        r"\bdate\b",
        r"\btime\b",
        r"\bpublished\b",
        r"\brecent\b",
        r"\brecently\b",
        r"\bmodified\b",
        r"\bupdated\b",
        r"\bcreated\b",
    ]

    list_patterns = [
        r"\blist\b",
        r"\bshow\b",
        r"\bgive me\b",
        r"\bexport\b",
        r"\bwhich\b",
    ]

    if any(re.search(pattern, q) for pattern in count_patterns):
        return "count"

    if any(re.search(pattern, q) for pattern in status_patterns):
        return "status_lookup"

    if any(re.search(pattern, q) for pattern in field_patterns):
        return "field_lookup"

    if any(re.search(pattern, q) for pattern in time_patterns):
        return "time_lookup"

    if any(re.search(pattern, q) for pattern in list_patterns):
        return "list"

    return "lookup"


def infer_domain(
    query: str,
    api_endpoints: List[str],
    sql_tables: List[str],
) -> str:
    q = query.lower()
    endpoint_text = " ".join(api_endpoints).lower()
    table_text = " ".join(sql_tables).lower()

    combined = f"{q} {endpoint_text} {table_text}"

    domain_rules = [
        ("segment_job", ["segment job", "segment jobs", "segment/jobs", "evaluation job"]),
        ("segment_definition", ["segment definition", "segment definitions", "segment/definitions"]),
        ("merge_policy", ["merge policy", "mergepolicies"]),
        ("journey", ["journey", "campaign", "ajo/journey", "dim_campaign"]),
        ("batch", ["batch", "batches", "catalog/batches"]),
        ("schema", ["schema", "schemas", "blueprint", "schemaregistry", "dim_blueprint"]),
        ("dataset", ["dataset", "datasets", "collection", "catalog/datasets", "dim_collection"]),
        ("destination", ["destination", "target", "dim_target"]),
        ("dataflow", ["dataflow", "flow", "flowservice/flows", "dim_connector"]),
        ("segment", ["segment", "audience", "audiences", "dim_segment"]),
        ("tag", ["tag", "tags", "unifiedtags"]),
        ("metric", ["metric", "metrics", "observability", "recordsuccess", "batchsuccess"]),
        ("property", ["field", "property", "attribute", "dim_property"]),
    ]

    scores: Dict[str, int] = {}

    for domain, keywords in domain_rules:
        score = 0

        for keyword in keywords:
            if keyword in combined:
                score += 1

        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    return max(scores.items(), key=lambda item: item[1])[0]


def build_search_text(
    query: str,
    route: str,
    intent: str,
    domain: str,
    sql_tables: List[str],
    api_endpoints: List[str],
    api_params: List[str],
) -> str:
    pieces = [
        query,
        f"route {route}",
        f"intent {intent}",
        f"domain {domain}",
        "tables " + " ".join(sql_tables),
        "endpoints " + " ".join(api_endpoints),
        "params " + " ".join(api_params),
    ]

    return " ".join(piece for piece in pieces if piece.strip())


def build_query_search_text(query: str) -> str:
    """
    Used at runtime for retrieval.
    We do not know route yet, but we can infer weak intent/domain signals.
    """
    intent = infer_intent(query)
    domain = infer_domain(query=query, api_endpoints=[], sql_tables=[])

    pieces = [
        query,
        f"intent {intent}",
        f"domain {domain}",
    ]

    return " ".join(pieces)


def build_example_index(examples: List[ExampleRecord]) -> List[IndexedExample]:
    indexed: List[IndexedExample] = []

    for example in examples:
        route = infer_route(example)
        sql_tables = extract_sql_tables(example.gold_sql)

        api_calls = [parse_gold_api(api) for api in example.gold_api]

        api_endpoints = sorted(
            set(call.endpoint for call in api_calls if call.endpoint.strip())
        )

        api_params = sorted(
            set(
                param
                for call in api_calls
                for param in call.params.keys()
            )
        )

        intent = infer_intent(example.query)
        domain = infer_domain(example.query, api_endpoints, sql_tables)

        search_text = build_search_text(
            query=example.query,
            route=route,
            intent=intent,
            domain=domain,
            sql_tables=sql_tables,
            api_endpoints=api_endpoints,
            api_params=api_params,
        )

        indexed.append(
            IndexedExample(
                query=example.query,
                route=route,
                intent=intent,
                domain=domain,
                sql_tables=sql_tables,
                api_calls=api_calls,
                api_endpoints=api_endpoints,
                api_params=api_params,
                gold_sql=example.gold_sql,
                gold_api=example.gold_api,
                search_text=search_text,
            )
        )

    return indexed


def save_example_index(
    indexed_examples: List[IndexedExample],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [asdict(example) for example in indexed_examples]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def build_tfidf_index(
    indexed_examples: List[IndexedExample],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [example.search_text for example in indexed_examples]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, output_dir / "tfidf_matrix.joblib")


def build_dense_embedding_index(
    indexed_examples: List[IndexedExample],
    output_dir: str | Path,
    model_name: str,
) -> None:
    """
    Builds dense semantic embeddings once and saves them.

    At runtime, we only embed the new query and compare it with this saved matrix.
    """
    from sentence_transformers import SentenceTransformer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [example.search_text for example in indexed_examples]

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding example texts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(output_dir / "dense_embeddings.npy", embeddings)


def save_retrieval_config(
    output_dir: str | Path,
    num_examples: int,
    dense_enabled: bool,
    embedding_model: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "num_examples": num_examples,
        "tfidf_enabled": True,
        "dense_enabled": dense_enabled,
        "embedding_model": embedding_model if dense_enabled else None,
        "description": "Hybrid retrieval index using TF-IDF, dense embeddings, and structured signals.",
    }

    with (output_dir / "retrieval_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def print_index_summary(indexed_examples: List[IndexedExample]) -> None:
    route_counts: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}
    intent_counts: Dict[str, int] = {}

    for example in indexed_examples:
        route_counts[example.route] = route_counts.get(example.route, 0) + 1
        domain_counts[example.domain] = domain_counts.get(example.domain, 0) + 1
        intent_counts[example.intent] = intent_counts.get(example.intent, 0) + 1

    print("\nRoute counts:")
    for route, count in sorted(route_counts.items()):
        print(f"  {route}: {count}")

    print("\nDomain counts:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    print("\nIntent counts:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/data.json")
    parser.add_argument("--out", default="data/example_index.json")
    parser.add_argument("--retrieval-dir", default="data/retrieval_index")

    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Dense embedding model from sentence-transformers.",
    )

    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense embedding index creation.",
    )

    args = parser.parse_args()

    examples = load_examples(args.data)
    indexed_examples = build_example_index(examples)

    save_example_index(indexed_examples, args.out)
    build_tfidf_index(indexed_examples, args.retrieval_dir)

    dense_enabled = not args.no_dense

    if dense_enabled:
        build_dense_embedding_index(
            indexed_examples=indexed_examples,
            output_dir=args.retrieval_dir,
            model_name=args.embedding_model,
        )

    save_retrieval_config(
        output_dir=args.retrieval_dir,
        num_examples=len(indexed_examples),
        dense_enabled=dense_enabled,
        embedding_model=args.embedding_model,
    )

    print(f"\nLoaded examples: {len(examples)}")
    print(f"Saved example index to: {args.out}")
    print(f"Saved retrieval index to: {args.retrieval_dir}")

    print_index_summary(indexed_examples)


if __name__ == "__main__":
    main()