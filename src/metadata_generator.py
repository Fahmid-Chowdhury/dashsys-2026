import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.query_analyzer import analyze_query


# ROUTE_BUDGETS = {
#     "API_ONLY": {
#         "max_sql_calls": 0,
#         "max_api_calls": 1,
#         "expected_steps": ["api_call"],
#     },
#     "SQL_ONLY": {
#         "max_sql_calls": 1,
#         "max_api_calls": 0,
#         "expected_steps": ["sql_query"],
#     },
#     "SQL_PLUS_API": {
#         "max_sql_calls": 1,
#         "max_api_calls": 1,
#         "expected_steps": ["sql_query", "api_call"],
#     },
#     "API_CHAIN": {
#         "max_sql_calls": 0,
#         "max_api_calls": 2,
#         "expected_steps": ["api_call", "api_call"],
#     },
#     "SQL_PLUS_API_CHAIN": {
#         "max_sql_calls": 1,
#         "max_api_calls": 2,
#         "expected_steps": ["sql_query", "api_call", "api_call"],
#     },
#     "UNKNOWN": {
#         "max_sql_calls": 1,
#         "max_api_calls": 1,
#         "expected_steps": ["sql_query", "api_call"],
#     },
# }


def build_tool_budget(route: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    complexity = analysis.get("complexity", "simple")
    subquery_count = len(analysis.get("subqueries", []))

    if complexity == "simple":
        if route == "SQL_ONLY":
            return {
                "max_sql_calls": 1,
                "max_api_calls": 0,
                "expected_steps": ["sql_query"],
            }

        if route == "API_ONLY":
            return {
                "max_sql_calls": 0,
                "max_api_calls": 1,
                "expected_steps": ["api_call"],
            }

        if route == "SQL_PLUS_API":
            return {
                "max_sql_calls": 1,
                "max_api_calls": 1,
                "expected_steps": ["sql_query", "api_call"],
            }

        if route == "API_CHAIN":
            return {
                "max_sql_calls": 0,
                "max_api_calls": 2,
                "expected_steps": ["api_call"],
            }

        if route == "SQL_PLUS_API_CHAIN":
            return {
                "max_sql_calls": 1,
                "max_api_calls": 2,
                "expected_steps": ["sql_query", "api_call"],
            }

    # Complex query budget.
    if route == "SQL_ONLY":
        return {
            "max_sql_calls": min(2, max(1, subquery_count)),
            "max_api_calls": 0,
            "expected_steps": ["sql_query"],
        }

    if route == "API_ONLY":
        return {
            "max_sql_calls": 0,
            "max_api_calls": min(3, max(1, subquery_count)),
            "expected_steps": ["api_call"],
        }

    if route == "SQL_PLUS_API":
        return {
            "max_sql_calls": 2,
            "max_api_calls": min(3, max(2, subquery_count)),
            "expected_steps": ["sql_query", "api_call"],
        }

    if route == "API_CHAIN":
        return {
            "max_sql_calls": 0,
            "max_api_calls": min(4, max(2, subquery_count)),
            "expected_steps": ["api_call"],
        }

    if route == "SQL_PLUS_API_CHAIN":
        return {
            "max_sql_calls": 2,
            "max_api_calls": min(4, max(2, subquery_count)),
            "expected_steps": ["sql_query", "api_call"],
        }

    return {
        "max_sql_calls": 2 if complexity == "complex" else 1,
        "max_api_calls": 2 if complexity == "complex" else 1,
        "expected_steps": [],
    }


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    return query


def extract_quoted_entities(query: str) -> List[str]:
    single = re.findall(r"'([^']+)'", query)
    double = re.findall(r'"([^"]+)"', query)
    return sorted(set(single + double))


def infer_basic_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["how many", "count", "number of", "total number"]):
        return "count"

    if any(x in q for x in ["failed", "success", "queued", "inactive", "active", "status"]):
        return "status_lookup"

    if any(x in q for x in ["field", "property", "attribute"]):
        return "field_lookup"

    if any(x in q for x in ["when", "published", "modified", "created", "updated"]):
        return "time_lookup"

    if any(x in q for x in ["list", "show", "give me", "export"]):
        return "list"

    return "lookup"


def extract_filters(query: str) -> Dict[str, Any]:
    q = query.lower()
    filters: Dict[str, Any] = {}

    status_values = [
        "success",
        "failed",
        "queued",
        "active",
        "inactive",
        "enabled",
        "disabled",
        "live",
        "published",
    ]

    for status in status_values:
        if status in q:
            filters["status_or_state"] = status
            break

    sandbox_match = re.search(r"\b([a-zA-Z0-9_-]+)\s+sandbox\b", query, re.IGNORECASE)
    if sandbox_match:
        filters["sandbox"] = sandbox_match.group(1)

    entities = extract_quoted_entities(query)
    if entities:
        filters["quoted_entities"] = entities

    return filters


def normalize_retrieval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": result.get("query", ""),
        "route": result.get("route", "UNKNOWN"),
        "domain": result.get("domain", "general"),
        "intent": result.get("intent", "lookup"),
        "combined_score": float(result.get("combined_score", result.get("score", 0.0))),
        "tfidf_score": float(result.get("tfidf_score", 0.0)),
        "dense_score": float(result.get("dense_score", 0.0)),
        "structured_score": float(result.get("structured_score", 0.0)),
        "sql_tables": result.get("sql_tables", []) or [],
        "api_endpoints": result.get("api_endpoints", []) or [],
        "api_params": result.get("api_params", []) or [],
        "gold_sql": result.get("gold_sql", "") or "",
        "gold_api": result.get("gold_api", []) or [],
    }
    

def should_force_sql_only(
    query: str,
    analysis: Dict[str, Any],
    route: str,
    domain: str,
    allowed_tables: Dict[str, Any],
) -> bool:
    if "SQL" not in route:
        return False

    if not allowed_tables:
        return False

    q = query.lower()
    complexity = analysis.get("complexity", "simple")
    purposes = {
        item.get("purpose")
        for item in analysis.get("subqueries", [])
        if isinstance(item, dict)
    }

    local_kg_domains = {
        "journey",
        "segment",
        "destination",
        "dataset",
        "schema",
        "property",
        "dataflow",
    }

    # Relationship/count/detail questions over KG snapshot should usually be SQL-only.
    if domain in local_kg_domains:
        if complexity == "complex":
            return True

        if purposes & {
            "relationship_lookup",
            "detail_lookup",
            "count_lookup",
            "entity_lookup",
        }:
            return True

    # Time lookup for a named journey/campaign can be answered from dim_campaign.
    if domain == "journey" and any(
        word in q
        for word in ["when", "published", "created", "updated", "modified"]
    ):
        return True

    return False


def endpoint_has_path_placeholder(endpoint_record: Dict[str, Any]) -> bool:
    path = endpoint_record.get("path", "") or ""
    return "{" in path and "}" in path


def create_hybrid_retriever(
    example_index_path: str = "data/example_index.json",
    retrieval_index_dir: str = "data/retrieval_index",
) -> Any:
    """
    Flexible loader so this works even if your HybridExampleRetriever constructor
    has slightly different parameter names.
    """

    from src.hybrid_retriever import HybridExampleRetriever

    signature = inspect.signature(HybridExampleRetriever)
    kwargs = {}

    if "example_index_path" in signature.parameters:
        kwargs["example_index_path"] = example_index_path

    if "retrieval_index_dir" in signature.parameters:
        kwargs["retrieval_index_dir"] = retrieval_index_dir
        
    if "retrieval_dir" in signature.parameters:
        kwargs["retrieval_dir"] = retrieval_index_dir

    if "index_dir" in signature.parameters:
        kwargs["index_dir"] = retrieval_index_dir

    if "data_dir" in signature.parameters:
        kwargs["data_dir"] = str(Path(example_index_path).parent)

    return HybridExampleRetriever(**kwargs)


def retrieve_similar_examples(
    query: str,
    top_k: int = 5,
    example_index_path: str = "data/example_index.json",
    retrieval_index_dir: str = "data/retrieval_index",
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    retriever = create_hybrid_retriever(
        example_index_path=example_index_path,
        retrieval_index_dir=retrieval_index_dir,
    )

    if hasattr(retriever, "retrieve"):
        raw_results = retriever.retrieve(query, top_k=top_k)
    elif hasattr(retriever, "search"):
        raw_results = retriever.search(query, top_k=top_k)
    else:
        raise AttributeError(
            "HybridExampleRetriever must have either retrieve() or search()."
        )

    results = [normalize_retrieval_result(r) for r in raw_results]

    if hasattr(retriever, "route_votes"):
        route_votes = retriever.route_votes(raw_results)
    else:
        route_votes = aggregate_votes(results, "route")

    if hasattr(retriever, "domain_votes"):
        domain_votes = retriever.domain_votes(raw_results)
    else:
        domain_votes = aggregate_votes(results, "domain")

    return results, route_votes, domain_votes


def search_with_existing_retriever(
    retriever: Any,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if hasattr(retriever, "retrieve"):
        raw_results = retriever.retrieve(query, top_k=top_k)
    elif hasattr(retriever, "search"):
        raw_results = retriever.search(query, top_k=top_k)
    else:
        raise AttributeError(
            "HybridExampleRetriever must have either retrieve() or search()."
        )

    return [normalize_retrieval_result(r) for r in raw_results]


def merge_retrieved_examples(
    results: List[Dict[str, Any]],
    max_examples: int = 8,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for item in results:
        key = item.get("query", "")

        if not key:
            continue

        score = float(item.get("combined_score", 0.0))

        if key not in merged:
            copied = dict(item)
            copied["merged_score"] = score
            copied["retrieval_sources"] = [item.get("retrieval_source", "unknown")]
            copied["matched_subqueries"] = []

            if item.get("subquery"):
                copied["matched_subqueries"].append(
                    {
                        "subquery": item.get("subquery"),
                        "purpose": item.get("purpose", "other"),
                        "score": score,
                    }
                )

            merged[key] = copied

        else:
            merged[key]["merged_score"] += score
            merged[key]["combined_score"] = max(
                float(merged[key].get("combined_score", 0.0)),
                score,
            )

            source = item.get("retrieval_source", "unknown")
            if source not in merged[key]["retrieval_sources"]:
                merged[key]["retrieval_sources"].append(source)

            if item.get("subquery"):
                merged[key]["matched_subqueries"].append(
                    {
                        "subquery": item.get("subquery"),
                        "purpose": item.get("purpose", "other"),
                        "score": score,
                    }
                )

    sorted_items = sorted(
        merged.values(),
        key=lambda x: x.get("merged_score", x.get("combined_score", 0.0)),
        reverse=True,
    )

    return sorted_items[:max_examples]


def retrieve_examples_for_analysis(
    query: str,
    analysis: Dict[str, Any],
    example_index_path: str = "data/example_index.json",
    retrieval_index_dir: str = "data/retrieval_index",
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    retriever = create_hybrid_retriever(
        example_index_path=example_index_path,
        retrieval_index_dir=retrieval_index_dir,
    )

    all_results: List[Dict[str, Any]] = []

    # Always retrieve using the original full query.
    original_results = search_with_existing_retriever(
        retriever=retriever,
        query=query,
        top_k=3,
    )

    for item in original_results:
        item["retrieval_source"] = "original_query"
        all_results.append(item)

    # For complex queries, retrieve for each LLM-generated subquery.
    if analysis.get("complexity") == "complex":
        for idx, subquery in enumerate(analysis.get("subqueries", []), start=1):
            sq = subquery.get("query", "").strip()

            if not sq:
                continue

            sub_results = search_with_existing_retriever(
                retriever=retriever,
                query=sq,
                top_k=2,
            )

            for item in sub_results:
                item["retrieval_source"] = f"subquery_{idx}"
                item["subquery"] = sq
                item["purpose"] = subquery.get("purpose", "other")
                all_results.append(item)

    max_examples = 8 if analysis.get("complexity") == "complex" else 3
    merged_examples = merge_retrieved_examples(
        all_results,
        max_examples=max_examples,
    )

    # Use merged_score for voting.
    vote_ready = []

    for item in merged_examples:
        copied = dict(item)
        copied["combined_score"] = float(
            item.get("merged_score", item.get("combined_score", 0.0))
        )
        vote_ready.append(copied)

    route_votes = aggregate_votes(vote_ready, "route")
    domain_votes = aggregate_votes(vote_ready, "domain")

    return merged_examples, route_votes, domain_votes


def aggregate_votes(results: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    votes: Dict[str, float] = {}

    for item in results:
        key = item.get(field, "UNKNOWN")
        score = float(item.get("combined_score", 0.0))

        votes[key] = votes.get(key, 0.0) + score

    total = sum(votes.values()) or 1.0

    return {
        key: round(value / total, 4)
        for key, value in sorted(votes.items(), key=lambda x: x[1], reverse=True)
    }


def select_top_vote(votes: Dict[str, float], default: str) -> str:
    if not votes:
        return default

    return max(votes.items(), key=lambda x: x[1])[0]


def choose_route(
    query: str,
    similar_examples: List[Dict[str, Any]],
    route_votes: Dict[str, float],
) -> Dict[str, Any]:
    selected_route = select_top_vote(route_votes, "UNKNOWN")

    q = query.lower()

    # Conservative overrides.
    # These domains are usually API-only/live API-centric.
    api_only_keywords = [
        "batch",
        "batches",
        "merge polic",
        "tag",
        "tags",
        "segment job",
        "evaluation job",
        "observability",
        "metric",
    ]

    if any(keyword in q for keyword in api_only_keywords):
        if selected_route in {"UNKNOWN", "SQL_PLUS_API"}:
            selected_route = "API_ONLY"

    # KG relationship questions usually need SQL plus API validation.
    sql_plus_api_keywords = [
        "journey",
        "campaign",
        "schema",
        "dataset",
        "destination",
        "dataflow",
        "field",
        "property",
        "connected to",
        "use the same schema",
    ]

    if any(keyword in q for keyword in sql_plus_api_keywords):
        if selected_route in {"UNKNOWN", "API_ONLY"}:
            selected_route = "SQL_PLUS_API"

    return {
        "selected": selected_route,
        "votes": route_votes,
        "reason": (
            "Selected from hybrid retrieval route votes with conservative "
            "domain keyword overrides."
        ),
    }


def choose_domain(
    query: str,
    domain_votes: Dict[str, float],
) -> Dict[str, Any]:
    selected_domain = select_top_vote(domain_votes, "general")

    return {
        "selected": selected_domain,
        "votes": domain_votes,
        "reason": "Selected from top retrieved examples and domain votes.",
    }


def compact_table_info(table_info: Dict[str, Any], max_columns: int = 40) -> Dict[str, Any]:
    columns = table_info.get("columns", [])[:max_columns]

    return {
        "kind": table_info.get("kind"),
        "domain": table_info.get("domain"),
        "row_count": table_info.get("row_count"),
        "id_columns": table_info.get("id_columns", []),
        "columns": [
            {
                "name": col.get("name"),
                "type": col.get("type"),
            }
            for col in columns
        ],
    }


def query_needs_relationships(query: str) -> bool:
    q = query.lower()

    relationship_keywords = [
        "connected to",
        "associated with",
        "linked to",
        "used by",
        "uses",
        "same schema",
        "mapped to",
        "belongs to",
        "destination named",
        "audiences connected",
    ]

    return any(keyword in q for keyword in relationship_keywords)


def select_allowed_tables(
    route: str,
    domain: str,
    similar_examples: List[Dict[str, Any]],
    schema_index: Dict[str, Any],
    query: str = "",
    max_tables: int = 5,
) -> Dict[str, Any]:
    if "SQL" not in route:
        return {}

    tables = schema_index.get("tables", {})
    candidate_names: List[str] = []

    # Use tables only from examples that match the selected domain.
    for example in similar_examples:
        if example.get("domain") != domain:
            continue

        for table in example.get("sql_tables", []):
            if table in tables and table not in candidate_names:
                candidate_names.append(table)

    # Only add fallback domain tables when no retrieved table exists
    # or the query clearly requires relationships.
    if not candidate_names or query_needs_relationships(query):
        for table_name, table_info in tables.items():
            if table_info.get("domain") == domain and table_name not in candidate_names:
                candidate_names.append(table_name)

    selected_names = candidate_names[:max_tables]

    return {
        table_name: compact_table_info(tables[table_name])
        for table_name in selected_names
        if table_name in tables
    }


def select_join_candidates(
    allowed_tables: Dict[str, Any],
    schema_index: Dict[str, Any],
    query: str = "",
    max_joins: int = 12,
) -> List[Dict[str, Any]]:
    if not query_needs_relationships(query):
        return []

    table_names = set(allowed_tables.keys())
    joins = []

    for join in schema_index.get("join_candidates", []):
        left_table = join.get("left_table")
        right_table = join.get("right_table")

        # Keep only joins where BOTH tables are allowed.
        if left_table in table_names and right_table in table_names:
            joins.append(join)

    joins = sorted(
        joins,
        key=lambda x: 0 if x.get("confidence") == "high" else 1,
    )

    return joins[:max_joins]


def select_allowed_api_endpoints(
    route: str,
    domain: str,
    similar_examples: List[Dict[str, Any]],
    api_index: Dict[str, Any],
    max_endpoints: int = 5,
) -> List[Dict[str, Any]]:
    if "API" not in route:
        return []

    endpoints = api_index.get("endpoints", {})
    selected_keys: List[str] = []

    # Use endpoints only from examples that match the selected domain.
    for example in similar_examples:
        if example.get("domain") != domain:
            continue

        for endpoint in example.get("api_endpoints", []):
            if endpoint in endpoints and endpoint not in selected_keys:
                selected_keys.append(endpoint)

    # Domain-matching fallback only.
    for endpoint, record in endpoints.items():
        if record.get("domain") == domain and endpoint not in selected_keys:
            selected_keys.append(endpoint)

    selected = []

    for endpoint in selected_keys:
        record = endpoints[endpoint]

        if endpoint_has_path_placeholder(record):
            continue

        selected.append(
            {
                "endpoint": record.get("endpoint"),
                "method": record.get("method"),
                "path": record.get("path"),
                "domain": record.get("domain"),
                "param_names": record.get("param_names", []),
                "observed_params": record.get("observed_params", {}),
                "routes_seen": record.get("routes_seen", []),
                "source": record.get("source", "unknown"),
                "openapi": record.get("openapi"),
            }
        )

        if len(selected) >= max_endpoints:
            break

    return selected


def build_similar_examples_for_metadata(
    similar_examples: List[Dict[str, Any]],
    max_examples: int = 8,
) -> List[Dict[str, Any]]:
    compact = []

    for example in similar_examples[:max_examples]:
        compact.append(
            {
                "query": example["query"],
                "route": example["route"],
                "domain": example["domain"],
                "intent": example["intent"],
                "score": round(
                    float(example.get("merged_score", example.get("combined_score", 0.0))),
                    4,
                ),
                "retrieval_sources": example.get("retrieval_sources", []),
                "matched_subqueries": example.get("matched_subqueries", []),
                "sql_tables": example["sql_tables"],
                "api_endpoints": example["api_endpoints"],
                "api_params": example["api_params"],
                "gold_sql": example["gold_sql"],
                "gold_api": example["gold_api"],
            }
        )

    return compact


def build_answer_constraints(intent: str, route: str) -> List[str]:
    constraints = [
        "Use only verified SQL/API outputs as evidence.",
        "Do not invent IDs, timestamps, counts, statuses, or names.",
        "If SQL and API disagree, mention the discrepancy clearly.",
    ]

    if intent == "count":
        constraints.append("For count questions, prefer explicit total/count fields when available.")

    if intent == "list":
        constraints.append("For list questions, return concise rows/items with relevant identifiers.")

    if "API" in route:
        constraints.append("If API returns an error or empty result, state that clearly.")

    return constraints


def generate_metadata(
    query: str,
    schema_index_path: str = "data/schema_index.json",
    # api_index_path: str = "data/api_index.json",
    api_index_path: str = "data/api_index_enriched.json",
    example_index_path: str = "data/example_index.json",
    retrieval_index_dir: str = "data/retrieval_index",
    top_k: int = 5,
) -> Dict[str, Any]:
    normalized_query = normalize_query(query)
    analysis = analyze_query(query)
    intent = infer_basic_intent(normalized_query)
    filters = extract_filters(normalized_query)
    entities = extract_quoted_entities(normalized_query)

    schema_index = load_json(schema_index_path)
    api_index = load_json(api_index_path)

    similar_examples, route_votes, domain_votes = retrieve_examples_for_analysis(
        query=normalized_query,
        analysis=analysis,
        example_index_path=example_index_path,
        retrieval_index_dir=retrieval_index_dir,
    )

    route_info = choose_route(
        query=normalized_query,
        similar_examples=similar_examples,
        route_votes=route_votes,
    )

    domain_info = choose_domain(
        query=normalized_query,
        domain_votes=domain_votes,
    )

    route = route_info["selected"]
    domain = domain_info["selected"]

    allowed_tables = select_allowed_tables(
        route=route,
        domain=domain,
        similar_examples=similar_examples,
        schema_index=schema_index,
        query=normalized_query,
    )
    
    if should_force_sql_only(
        query=normalized_query,
        analysis=analysis,
        route=route,
        domain=domain,
        allowed_tables=allowed_tables,
    ):
        route = "SQL_ONLY"
        route_info["selected"] = "SQL_ONLY"
        route_info["reason"] += " Forced to SQL_ONLY because query analysis indicates local KG evidence is sufficient."

    join_candidates = select_join_candidates(
        allowed_tables=allowed_tables,
        schema_index=schema_index,
        query=normalized_query,
    )

    allowed_api_endpoints = select_allowed_api_endpoints(
        route=route,
        domain=domain,
        similar_examples=similar_examples,
        api_index=api_index,
    )

    tool_budget = build_tool_budget(route, analysis)

    metadata = {
        "query": query,
        "normalized_query": normalized_query,
        "query_analysis": analysis,
        "intent": intent,
        "entities": entities,
        "filters": filters,
        "route": route_info,
        "domain": domain_info,
        "allowed_tables": allowed_tables,
        "join_candidates": join_candidates,
        "allowed_api_endpoints": allowed_api_endpoints,
        "similar_examples": build_similar_examples_for_metadata(
            similar_examples,
            max_examples=8 if analysis.get("complexity") == "complex" else 3,
        ),
        "tool_budget": tool_budget,
        "answer_constraints": build_answer_constraints(intent, route),
    }

    return metadata


def print_metadata_summary(metadata: Dict[str, Any]) -> None:
    print("Metadata summary")
    print("================")
    print("Query:", metadata["query"])
    print("Route:", metadata["route"]["selected"])
    print("Domain:", metadata["domain"]["selected"])
    print("Intent:", metadata["intent"])

    print("\nAllowed tables:")
    if metadata["allowed_tables"]:
        for table in metadata["allowed_tables"]:
            print(f"  - {table}")
    else:
        print("  none")

    print("\nAllowed API endpoints:")
    if metadata["allowed_api_endpoints"]:
        for endpoint in metadata["allowed_api_endpoints"]:
            print(f"  - {endpoint['endpoint']}")
    else:
        print("  none")

    print("\nSimilar examples:")
    for ex in metadata["similar_examples"]:
        print(f"  - [{ex['score']}] {ex['query']}")


if __name__ == "__main__":
    sample_query = "List all journeys"

    metadata = generate_metadata(sample_query)
    save_json(metadata, "data/sample_metadata.json")

    print_metadata_summary(metadata)
    print("\nSaved to: data/sample_metadata.json")