import argparse
import csv
import inspect
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

from src.agent_runner import (
    AgentRunConfig,
    create_llm_client,
    create_answer_llm_client,
    TraceRouterAgent,
    save_agent_run,
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("MODEL", "gemma4:latest")

ANSWER_LLM_PROVIDER = os.getenv("ANSWER_LLM_PROVIDER", LLM_PROVIDER)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", MODEL)

print(f"Using action provider: {LLM_PROVIDER}")
print(f"Using action model: {MODEL}")
print(f"Using answer provider: {ANSWER_LLM_PROVIDER}")
print(f"Using answer model: {ANSWER_MODEL}")

@dataclass
class BatchRunConfig:
    test_path: str = "data/test.json"
    output_dir: str = "data/test_runs"
    metadata_output_dir: str = "data/test_metadata"
    summary_path: str = "data/test_runs_summary.json"
    metrics_path: str = "data/metrics/test_runs_metrics.json"
    metrics_csv_path: str = "data/metrics/test_runs_metrics.csv"

    example_index_path: str = "data/example_index.json"
    retrieval_index_dir: str = "data/retrieval_index"
    schema_index_path: str = "data/schema_index.json"
    api_index_path: str = "data/api_index_enriched.json"
    
    llm_provider: str = LLM_PROVIDER
    model: str = MODEL
    
    answer_llm_provider: str = ANSWER_LLM_PROVIDER
    answer_model: str = ANSWER_MODEL
    
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


def count_successful_trace_action(trace: List[Dict[str, Any]], action_name: str) -> int:
    """
    Counts only executed tool calls that succeeded.

    SQL stores success directly on the step.
    API stores success inside step["api_call"].
    Mock API success is counted as success.
    """

    count = 0

    for step in trace:
        if step.get("action") != action_name:
            continue

        if action_name == "sql_query" and step.get("status") == "success":
            count += 1

        elif action_name == "api_call":
            api_call = step.get("api_call", {}) or {}
            if api_call.get("status") in {"success", "mock_success"}:
                count += 1

    return count


def get_tool_budget(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the route-aware per-query budget from metadata.
    """

    budget = metadata.get("tool_budget", {}) or {}
    expected_steps = budget.get("expected_steps", []) or []

    if not isinstance(expected_steps, list):
        expected_steps = []

    return {
        "max_sql_calls": int(budget.get("max_sql_calls", 0) or 0),
        "max_api_calls": int(budget.get("max_api_calls", 0) or 0),
        "expected_steps": [
            step for step in expected_steps
            if step in {"sql_query", "api_call"}
        ],
    }


def count_expected_tool_actions(expected_steps: List[str]) -> Counter:
    return Counter(
        step for step in expected_steps
        if step in {"sql_query", "api_call"}
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0

    return round(float(numerator) / float(denominator), 4)


def summarize_budget_usage(
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
    config: BatchRunConfig,
) -> Dict[str, Any]:
    """
    Paper-ready per-query budget summary.

    Records:
    - allowed budget
    - used budget
    - remaining budget
    - budget usage ratios
    - expected SQL/API evidence satisfaction
    """

    budget = get_tool_budget(metadata)
    expected_counts = count_expected_tool_actions(budget["expected_steps"])

    sql_calls = count_trace_action(trace, "sql_query")
    api_calls = count_trace_action(trace, "api_call")

    successful_sql_calls = count_successful_trace_action(trace, "sql_query")
    successful_api_calls = count_successful_trace_action(trace, "api_call")

    max_sql_calls = budget["max_sql_calls"]
    max_api_calls = budget["max_api_calls"]
    max_total_tool_calls = max_sql_calls + max_api_calls

    expected_sql_calls = expected_counts.get("sql_query", 0)
    expected_api_calls = expected_counts.get("api_call", 0)

    return {
        "allowed": {
            "max_sql_calls": max_sql_calls,
            "max_api_calls": max_api_calls,
            "max_total_tool_calls": max_total_tool_calls,
            "max_agent_steps": config.max_steps,
            "max_repair_attempts": config.max_repair_attempts,
            "expected_steps": budget["expected_steps"],
            "expected_sql_calls": expected_sql_calls,
            "expected_api_calls": expected_api_calls,
        },
        "used": {
            "sql_calls": sql_calls,
            "api_calls": api_calls,
            "total_tool_calls": sql_calls + api_calls,
            "successful_sql_calls": successful_sql_calls,
            "successful_api_calls": successful_api_calls,
            "successful_total_tool_calls": successful_sql_calls + successful_api_calls,
            "trace_steps": len(trace),
        },
        "remaining": {
            "sql_calls": max(max_sql_calls - sql_calls, 0),
            "api_calls": max(max_api_calls - api_calls, 0),
            "total_tool_calls": max(max_total_tool_calls - (sql_calls + api_calls), 0),
            "agent_steps": max(config.max_steps - len(trace), 0),
        },
        "ratios": {
            "sql_budget_used_ratio": safe_ratio(sql_calls, max_sql_calls),
            "api_budget_used_ratio": safe_ratio(api_calls, max_api_calls),
            "total_tool_budget_used_ratio": safe_ratio(sql_calls + api_calls, max_total_tool_calls),
            "agent_step_used_ratio": safe_ratio(len(trace), config.max_steps),
        },
        "flags": {
            "within_sql_budget": sql_calls <= max_sql_calls,
            "within_api_budget": api_calls <= max_api_calls,
            "within_total_tool_budget": (sql_calls + api_calls) <= max_total_tool_calls,
            "expected_sql_satisfied": successful_sql_calls >= expected_sql_calls,
            "expected_api_satisfied": successful_api_calls >= expected_api_calls,
            "expected_tool_steps_satisfied": (
                successful_sql_calls >= expected_sql_calls
                and successful_api_calls >= expected_api_calls
            ),
        },
    }


def infer_failure_reason(summary: Dict[str, Any]) -> str:
    """
    Produces one compact failure label for aggregate failure-rate reporting.
    """

    if summary.get("status") == "success":
        return "none"

    if summary.get("status") == "failed_exception":
        return "exception"

    debug_counts = summary.get("debug_event_counts", {}) or {}

    for event_name in [
        "tool_failure_detected_before_final_answer",
        "answer_agent_failed",
        "json_parse_failed",
        "route_policy_rejected",
        "budget_rejected",
        "verifier_rejected",
    ]:
        if debug_counts.get(event_name, 0) > 0:
            return event_name

    if summary.get("status") == "failed":
        return "max_steps_or_unresolved"

    return str(summary.get("status", "unknown"))


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
    config: BatchRunConfig,
) -> Dict[str, Any]:
    trace = result.get("trace", []) or []
    debug_events = result.get("debug_events", []) or []

    budget_usage = summarize_budget_usage(
        metadata=metadata,
        trace=trace,
        config=config,
    )

    summary = {
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
        "per_query_budget": budget_usage,
        "answer_preview": str(result.get("answer", ""))[:400],
        "elapsed_seconds": round(elapsed_seconds, 3),
    }

    summary["failure_reason"] = infer_failure_reason(summary)

    return summary


def build_batch_metrics(
    summaries: List[Dict[str, Any]],
    batch_wall_time_seconds: float,
) -> Dict[str, Any]:
    """
    Builds the three paper-focused metric groups:
    1. Wall time
    2. Per-query budget usage
    3. Failure rate
    """

    total_queries = len(summaries)
    success_count = sum(1 for item in summaries if item.get("status") == "success")
    failure_count = total_queries - success_count

    elapsed_values = [
        float(item.get("elapsed_seconds", 0.0) or 0.0)
        for item in summaries
    ]

    sql_calls = [int(item.get("sql_calls", 0) or 0) for item in summaries]
    api_calls = [int(item.get("api_calls", 0) or 0) for item in summaries]
    trace_steps = [int(item.get("trace_steps", 0) or 0) for item in summaries]

    statuses = Counter(str(item.get("status", "unknown")) for item in summaries)

    failure_reasons = Counter(
        item.get("failure_reason", "unknown")
        for item in summaries
        if item.get("status") != "success"
    )

    budget_flags = [
        ((item.get("per_query_budget", {}) or {}).get("flags", {}) or {})
        for item in summaries
    ]

    within_tool_budget = sum(
        1 for flags in budget_flags
        if (
            flags.get("within_sql_budget") is True
            and flags.get("within_api_budget") is True
            and flags.get("within_total_tool_budget") is True
        )
    )

    expected_tool_steps_satisfied = sum(
        1 for flags in budget_flags
        if flags.get("expected_tool_steps_satisfied") is True
    )

    def avg(values: Sequence[int | float]) -> float:
        return round(mean(values), 3) if values else 0.0

    def med(values: Sequence[int | float]) -> float:
        return round(median(values), 3) if values else 0.0

    return {
        "total_queries": total_queries,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": safe_ratio(failure_count, total_queries),
        "success_rate": safe_ratio(success_count, total_queries),
        "status_counts": dict(statuses),
        "failure_reason_counts": dict(failure_reasons),
        "wall_time": {
            "batch_wall_time_seconds": round(batch_wall_time_seconds, 3),
            "sum_query_elapsed_seconds": round(sum(elapsed_values), 3),
            "avg_query_elapsed_seconds": avg(elapsed_values),
            "median_query_elapsed_seconds": med(elapsed_values),
            "min_query_elapsed_seconds": round(min(elapsed_values), 3) if elapsed_values else 0.0,
            "max_query_elapsed_seconds": round(max(elapsed_values), 3) if elapsed_values else 0.0,
        },
        "budget": {
            "within_tool_budget_count": within_tool_budget,
            "within_tool_budget_rate": safe_ratio(within_tool_budget, total_queries),
            "expected_tool_steps_satisfied_count": expected_tool_steps_satisfied,
            "expected_tool_steps_satisfied_rate": safe_ratio(expected_tool_steps_satisfied, total_queries),
            "avg_sql_calls_per_query": avg(sql_calls),
            "avg_api_calls_per_query": avg(api_calls),
            "avg_total_tool_calls_per_query": avg([
                sql + api for sql, api in zip(sql_calls, api_calls)
            ]),
            "avg_trace_steps_per_query": avg(trace_steps),
        },
        "per_query": summaries,
    }


def flatten_summary_for_csv(summary: Dict[str, Any]) -> Dict[str, Any]:
    budget = summary.get("per_query_budget", {}) or {}
    allowed = budget.get("allowed", {}) or {}
    used = budget.get("used", {}) or {}
    remaining = budget.get("remaining", {}) or {}
    ratios = budget.get("ratios", {}) or {}
    flags = budget.get("flags", {}) or {}

    return {
        "id": summary.get("id"),
        "query": summary.get("query"),
        "status": summary.get("status"),
        "failure_reason": summary.get("failure_reason"),
        "route": summary.get("route"),
        "domain": summary.get("domain"),
        "intent": summary.get("intent"),
        "complexity": summary.get("complexity"),
        "elapsed_seconds": summary.get("elapsed_seconds"),

        "max_sql_calls": allowed.get("max_sql_calls"),
        "max_api_calls": allowed.get("max_api_calls"),
        "max_total_tool_calls": allowed.get("max_total_tool_calls"),
        "expected_sql_calls": allowed.get("expected_sql_calls"),
        "expected_api_calls": allowed.get("expected_api_calls"),

        "sql_calls": used.get("sql_calls"),
        "api_calls": used.get("api_calls"),
        "total_tool_calls": used.get("total_tool_calls"),
        "successful_sql_calls": used.get("successful_sql_calls"),
        "successful_api_calls": used.get("successful_api_calls"),

        "remaining_sql_calls": remaining.get("sql_calls"),
        "remaining_api_calls": remaining.get("api_calls"),

        "sql_budget_used_ratio": ratios.get("sql_budget_used_ratio"),
        "api_budget_used_ratio": ratios.get("api_budget_used_ratio"),
        "total_tool_budget_used_ratio": ratios.get("total_tool_budget_used_ratio"),

        "within_sql_budget": flags.get("within_sql_budget"),
        "within_api_budget": flags.get("within_api_budget"),
        "within_total_tool_budget": flags.get("within_total_tool_budget"),
        "expected_sql_satisfied": flags.get("expected_sql_satisfied"),
        "expected_api_satisfied": flags.get("expected_api_satisfied"),
        "expected_tool_steps_satisfied": flags.get("expected_tool_steps_satisfied"),
    }


def save_metrics_csv(summaries: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [flatten_summary_for_csv(summary) for summary in summaries]

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        

def create_answer_llm_client_safely(
    provider: str,
    model: str,
    temperature: float = 0.0,
):
    """
    Creates the answer-agent LLM.

    The answer agent should NOT be forced to use the SQL/API action schema.
    If create_llm_client supports use_json_schema, disable it.
    If not, fall back to the normal signature.
    """

    try:
        return create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    except TypeError:
        return create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
        )


def run_batch(config: BatchRunConfig) -> List[Dict[str, Any]]:
    batch_start_time = time.time()
    queries = load_test_queries(config.test_path)

    if config.start:
        queries = queries[config.start:]

    if config.limit is not None:
        queries = queries[:config.limit]

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    llm = create_llm_client(
        provider=config.llm_provider,
        model=config.model,
        temperature=0.0,
    )

    answer_llm = create_answer_llm_client_safely(
        provider=config.answer_llm_provider,
        model=config.answer_model,
        temperature=0.0,
    )

    agent_config = AgentRunConfig(
        max_steps=config.max_steps,
        max_repair_attempts=config.max_repair_attempts,
        snapshot_dir="DBSnapshot",
        api_index_path=config.api_index_path,
        mock_api=config.mock_api,
        save_prompts=True,
        use_answer_builder=True,
        use_answer_agent=True,
    )

    agent = TraceRouterAgent(
        llm_client=llm,
        answer_llm_client=answer_llm,
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
                config=config,
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
            
            summary["failure_reason"] = infer_failure_reason(summary)

            summaries.append(summary)

            error_path = Path(config.output_dir) / f"run_{query_id:03d}_error.json"
            save_json(summary, error_path)

            print(f"FAILED: {e}")
            print(f"Saved error: {error_path}")

        save_json(summaries, config.summary_path)

        partial_metrics = build_batch_metrics(
            summaries=summaries,
            batch_wall_time_seconds=time.time() - batch_start_time,
        )

        save_json(partial_metrics, config.metrics_path)
        save_metrics_csv(summaries, config.metrics_csv_path)

    print()
    print("Batch run complete.")
    print(f"Summary saved to: {config.summary_path}")

    final_metrics = build_batch_metrics(
        summaries=summaries,
        batch_wall_time_seconds=time.time() - batch_start_time,
    )

    save_json(final_metrics, config.metrics_path)
    save_metrics_csv(summaries, config.metrics_csv_path)

    print()
    print("Batch run complete.")
    print(f"Summary saved to: {config.summary_path}")
    print(f"Metrics saved to: {config.metrics_path}")
    print(f"Metrics CSV saved to: {config.metrics_csv_path}")
    print(
        "Failure rate: "
        f"{final_metrics['failure_rate']:.4f} "
        f"({final_metrics['failure_count']}/{final_metrics['total_queries']})"
    )
    print(
        "Batch wall time: "
        f"{final_metrics['wall_time']['batch_wall_time_seconds']} seconds"
    )

    return summaries


def parse_args() -> BatchRunConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/test.json")
    parser.add_argument("--output-dir", default="data/test_runs")
    parser.add_argument("--metadata-output-dir", default="data/test_metadata")
    parser.add_argument("--summary-path", default="data/test_runs_summary.json")

    parser.add_argument("--llm-provider", default=LLM_PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)

    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--answer-llm-provider", default=ANSWER_LLM_PROVIDER)
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--metrics-path", default="data/metrics/test_runs_metrics.json")
    parser.add_argument("--metrics-csv-path", default="data/metrics/test_runs_metrics.csv")

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
        metrics_path=args.metrics_path,
        metrics_csv_path=args.metrics_csv_path,
        llm_provider=args.llm_provider,
        model=args.model,
        answer_llm_provider=args.answer_llm_provider,
        answer_model=args.answer_model,
        limit=args.limit,
        start=args.start,
        max_steps=args.max_steps,
        max_repair_attempts=args.max_repair_attempts,
        mock_api=not args.real_api,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_batch(cfg)