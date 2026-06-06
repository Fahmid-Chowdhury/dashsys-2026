import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent_runner import (
    AgentRunConfig,
    OllamaLLMClient,
    TraceRouterAgent,
    save_agent_run,
)

MODEL = os.getenv("MODEL", "gemma4:latest")
print(f"Using model: {MODEL}")

@dataclass
class BatchRunConfig:
    test_path: str = "data/test.json"
    output_dir: str = "data/test_runs"
    metadata_output_dir: str = "data/test_metadata"
    summary_path: str = "data/test_runs_summary.json"

    example_index_path: str = "data/example_index.json"
    retrieval_index_dir: str = "data/retrieval_index"
    schema_index_path: str = "data/schema_index.json"
    api_index_path: str = "data/api_index_enriched.json"

    model: str = MODEL
    max_steps: int = 8
    max_repair_attempts: int = 3
    mock_api: bool = True
    limit: Optional[int] = None
    start: int = 0


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


def load_test_queries(test_path: str | Path) -> List[Dict[str, Any]]:
    data = load_json(test_path)

    if not isinstance(data, list):
        raise ValueError("test.json must contain a list of query objects.")

    queries = []

    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            queries.append({"id": idx, "query": item})
        elif isinstance(item, dict) and item.get("query"):
            q = dict(item)
            q.setdefault("id", idx)
            queries.append(q)
        else:
            raise ValueError(f"Invalid query object at index {idx}: {item}")

    return queries


def generate_metadata_for_query(query: str, config: BatchRunConfig) -> Dict[str, Any]:
    """
    Flexible wrapper around your existing src.metadata_generator.generate_metadata.

    This is intentionally defensive because your generate_metadata function
    may have slightly different parameters depending on your current version.
    """

    from src import metadata_generator

    generate_metadata = metadata_generator.generate_metadata
    signature = inspect.signature(generate_metadata)
    params = signature.parameters

    kwargs = {}

    if "query" in params:
        kwargs["query"] = query
    elif "user_query" in params:
        kwargs["user_query"] = query
    elif "natural_language_query" in params:
        kwargs["natural_language_query"] = query

    possible_paths = {
        "example_index_path": config.example_index_path,
        "retrieval_index_dir": config.retrieval_index_dir,
        "schema_index_path": config.schema_index_path,
        "api_index_path": config.api_index_path,
    }

    for key, value in possible_paths.items():
        if key in params:
            kwargs[key] = value

    try:
        if kwargs:
            metadata = generate_metadata(**kwargs)
        else:
            metadata = generate_metadata(query)

    except TypeError:
        metadata = generate_metadata(query)

    if not isinstance(metadata, dict):
        raise ValueError("generate_metadata did not return a dictionary.")

    metadata.setdefault("query", query)

    return metadata


def get_route(metadata: Dict[str, Any]) -> str:
    route = metadata.get("route", {})

    if isinstance(route, dict):
        return route.get("selected", "unknown")

    return str(route)


def get_domain(metadata: Dict[str, Any]) -> str:
    domain = metadata.get("domain", {})

    if isinstance(domain, dict):
        return domain.get("selected", "unknown")

    return str(domain)


def count_trace_action(trace: List[Dict[str, Any]], action_name: str) -> int:
    return sum(1 for step in trace if step.get("action") == action_name)


def summarize_debug_events(debug_events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for event in debug_events:
        name = event.get("event", "unknown")
        counts[name] = counts.get(name, 0) + 1

    return counts


def summarize_run(
    query_id: int,
    query: str,
    metadata: Dict[str, Any],
    result: Dict[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    trace = result.get("trace", []) or []
    debug_events = result.get("debug_events", []) or []

    return {
        "id": query_id,
        "query": query,
        "status": result.get("status"),
        "route": get_route(metadata),
        "domain": get_domain(metadata),
        "intent": metadata.get("intent", "unknown"),
        "complexity": (metadata.get("query_analysis", {}) or {}).get("complexity"),
        "subquery_count": len((metadata.get("query_analysis", {}) or {}).get("subqueries", [])),
        "sql_calls": count_trace_action(trace, "sql_query"),
        "api_calls": count_trace_action(trace, "api_call"),
        "trace_steps": len(trace),
        "debug_event_counts": summarize_debug_events(debug_events),
        "answer_preview": str(result.get("answer", ""))[:400],
        "elapsed_seconds": round(elapsed_seconds, 3),
    }

def run_batch(config: BatchRunConfig) -> List[Dict[str, Any]]:
    queries = load_test_queries(config.test_path)

    if config.start:
        queries = queries[config.start:]

    if config.limit is not None:
        queries = queries[:config.limit]

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    llm = OllamaLLMClient(
        model=config.model,
        temperature=0.0,
        num_ctx=8192,
        use_json_schema=True,
    )

    agent_config = AgentRunConfig(
        max_steps=config.max_steps,
        max_repair_attempts=config.max_repair_attempts,
        snapshot_dir="DBSnapshot",
        api_index_path=config.api_index_path,
        mock_api=config.mock_api,
        save_prompts=True,
        use_answer_builder=True,
    )

    agent = TraceRouterAgent(
        llm_client=llm,
        config=agent_config,
    )

    summaries: List[Dict[str, Any]] = []

    for local_idx, item in enumerate(queries, start=1):
        query_id = int(item.get("id", local_idx))
        query = item["query"]

        print()
        print("=" * 80)
        print(f"Running query {query_id}: {query}")
        print("=" * 80)

        start_time = time.time()

        try:
            metadata = generate_metadata_for_query(query, config)

            metadata_path = Path(config.metadata_output_dir) / f"metadata_{query_id:03d}.json"
            save_json(metadata, metadata_path)

            result = agent.run(
                query=query,
                metadata=metadata,
                run_id=f"test_{query_id:03d}",
            )

            run_path = Path(config.output_dir) / f"run_{query_id:03d}.json"
            save_agent_run(result, run_path)

            elapsed = time.time() - start_time

            summary = summarize_run(
                query_id=query_id,
                query=query,
                metadata=metadata,
                result=result,
                elapsed_seconds=elapsed,
            )

            summary["metadata_path"] = str(metadata_path)
            summary["run_path"] = str(run_path)

            summaries.append(summary)

            print(f"Status: {summary['status']}")
            print(f"Route: {summary['route']}")
            print(f"Domain: {summary['domain']}")
            print(f"SQL calls: {summary['sql_calls']}")
            print(f"API calls: {summary['api_calls']}")
            print(f"Saved: {run_path}")

        except Exception as e:
            elapsed = time.time() - start_time

            summary = {
                "id": query_id,
                "query": query,
                "status": "failed_exception",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 3),
            }

            summaries.append(summary)

            error_path = Path(config.output_dir) / f"run_{query_id:03d}_error.json"
            save_json(summary, error_path)

            print(f"FAILED: {e}")
            print(f"Saved error: {error_path}")

        save_json(summaries, config.summary_path)

    print()
    print("Batch run complete.")
    print(f"Summary saved to: {config.summary_path}")

    return summaries


def parse_args() -> BatchRunConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/test.json")
    parser.add_argument("--output-dir", default="data/test_runs")
    parser.add_argument("--metadata-output-dir", default="data/test_metadata")
    parser.add_argument("--summary-path", default="data/test_runs_summary.json")

    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)

    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-repair-attempts", type=int, default=3)

    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use real Adobe API calls. By default, API calls are mocked.",
    )

    args = parser.parse_args()

    return BatchRunConfig(
        test_path=args.input,
        output_dir=args.output_dir,
        metadata_output_dir=args.metadata_output_dir,
        summary_path=args.summary_path,
        model=args.model,
        limit=args.limit,
        start=args.start,
        max_steps=args.max_steps,
        max_repair_attempts=args.max_repair_attempts,
        mock_api=not args.real_api,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)