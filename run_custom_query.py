# run_custom_query.py

import argparse
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from src.agent_runner import (
    AgentRunConfig,
    TraceRouterAgent,
    create_llm_client,
    create_answer_llm_client,
    save_agent_run,
)

from src.prompt_builder import build_prompt_messages


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("MODEL", "gemma4:latest")

ANSWER_LLM_PROVIDER = os.getenv("ANSWER_LLM_PROVIDER", LLM_PROVIDER)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", MODEL)


@dataclass
class CustomRunConfig:
    output_dir: str = "custom_query_runs"

    example_index_path: str = "data/example_index.json"
    retrieval_index_dir: str = "data/retrieval_index"
    schema_index_path: str = "data/schema_index.json"
    api_index_path: str = "data/api_index_enriched.json"

    snapshot_dir: str = "DBSnapshot"

    llm_provider: str = LLM_PROVIDER
    model: str = MODEL

    answer_llm_provider: str = ANSWER_LLM_PROVIDER
    answer_model: str = ANSWER_MODEL

    max_steps: int = 8
    max_repair_attempts: int = 3

    # True = use mock API responses.
    # False = call real Adobe APIs using .env credentials.
    mock_api: bool = True


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "query"


def get_next_query_id(output_dir: str | Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_id = 0

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue

        match = re.match(r"query_(\d+)_", child.name)
        if match:
            max_id = max(max_id, int(match.group(1)))

    return max_id + 1


def create_query_folder(query: str, config: CustomRunConfig) -> Path:
    query_id = get_next_query_id(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(query)

    run_dir = Path(config.output_dir) / f"query_{query_id:03d}_{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def generate_metadata_for_query(query: str, config: CustomRunConfig) -> Dict[str, Any]:
    """
    Flexible wrapper around src.metadata_generator.generate_metadata.

    This avoids breaking if your generate_metadata function signature changes slightly.
    """

    from src import metadata_generator

    generate_metadata = metadata_generator.generate_metadata
    signature = inspect.signature(generate_metadata)
    params = signature.parameters

    kwargs: Dict[str, Any] = {}

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

    if kwargs:
        metadata = generate_metadata(**kwargs)
    else:
        metadata = generate_metadata(query)

    if not isinstance(metadata, dict):
        raise ValueError("generate_metadata did not return a dictionary.")

    metadata.setdefault("query", query)
    return metadata


def build_agent_config(config: CustomRunConfig, run_dir: Path) -> AgentRunConfig:
    """
    Creates AgentRunConfig safely.

    If your AgentRunConfig changes later, this only passes fields that exist.
    """

    desired_kwargs = {
        "max_steps": config.max_steps,
        "max_repair_attempts": config.max_repair_attempts,
        "snapshot_dir": config.snapshot_dir,
        "api_index_path": config.api_index_path,
        "mock_api": config.mock_api,
        "save_prompts": True,
        "prompt_output_dir": str(run_dir / "step_prompts"),
        "use_answer_builder": True,
        "use_answer_agent": True,
    }

    signature = inspect.signature(AgentRunConfig)
    valid_kwargs = {
        key: value
        for key, value in desired_kwargs.items()
        if key in signature.parameters
    }

    return AgentRunConfig(**valid_kwargs)


def create_agent(config: CustomRunConfig, run_dir: Path) -> TraceRouterAgent:
    action_llm = create_llm_client(
        provider=config.llm_provider,
        model=config.model,
        temperature=0.0,
    )

    answer_llm = create_answer_llm_client(
        provider=config.answer_llm_provider,
        model=config.answer_model,
        temperature=0.0,
    )

    agent_config = build_agent_config(config, run_dir)

    signature = inspect.signature(TraceRouterAgent)
    kwargs: Dict[str, Any] = {
        "llm_client": action_llm,
        "config": agent_config,
    }

    if "answer_llm_client" in signature.parameters:
        kwargs["answer_llm_client"] = answer_llm

    return TraceRouterAgent(**kwargs)


def save_filled_prompt_artifacts(
    query: str,
    metadata: Dict[str, Any],
    run_dir: Path,
) -> List[Dict[str, str]]:
    """
    Saves the initial filled prompt before any SQL/API tool result exists.

    This is the query-specific prompt that contains:
    - system instruction
    - user query
    - selected metadata
    - selected schema/API context
    - route/budget guidance
    """

    messages = build_prompt_messages(
        query=query,
        metadata=metadata,
        tool_results=None,
        verifier_feedback=None,
    )

    save_json(messages, run_dir / "filled_prompt_messages.json")

    readable_parts = []

    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        readable_parts.append(f"[{role}]\n{content}")

    save_text(
        "\n\n" + ("=" * 100) + "\n\n".join(readable_parts),
        run_dir / "filled_prompt.txt",
    )

    return messages


def summarize_result(
    query: str,
    metadata: Dict[str, Any],
    result: Dict[str, Any],
    elapsed_seconds: float,
    run_dir: Path,
) -> Dict[str, Any]:
    trace = result.get("trace", []) or []

    sql_calls = sum(1 for step in trace if step.get("action") == "sql_query")
    api_calls = sum(1 for step in trace if step.get("action") == "api_call")

    route = metadata.get("route", {})
    if isinstance(route, dict):
        route = route.get("selected", "unknown")

    domain = metadata.get("domain", {})
    if isinstance(domain, dict):
        domain = domain.get("selected", "unknown")

    return {
        "query": query,
        "status": result.get("status"),
        "route": route,
        "domain": domain,
        "intent": metadata.get("intent", "unknown"),
        "sql_calls": sql_calls,
        "api_calls": api_calls,
        "trace_steps": len(trace),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "answer": result.get("answer", ""),
        "folder": str(run_dir),
        "files": {
            "metadata": str(run_dir / "metadata.json"),
            "filled_prompt_messages": str(run_dir / "filled_prompt_messages.json"),
            "filled_prompt_text": str(run_dir / "filled_prompt.txt"),
            "agent_output_trajectory": str(run_dir / "agent_output_trajectory.json"),
            "trace_only": str(run_dir / "trace_only.json"),
            "answer": str(run_dir / "answer.txt"),
            "summary": str(run_dir / "summary.json"),
            "step_prompts": str(run_dir / "step_prompts"),
        },
    }


def run_one_query(query: str, config: CustomRunConfig) -> Dict[str, Any]:
    run_dir = create_query_folder(query, config)

    print()
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Output folder: {run_dir}")
    print("=" * 100)

    start_time = time.time()

    try:
        save_text(query, run_dir / "query.txt")

        metadata = generate_metadata_for_query(query, config)
        save_json(metadata, run_dir / "metadata.json")

        save_filled_prompt_artifacts(
            query=query,
            metadata=metadata,
            run_dir=run_dir,
        )

        agent = create_agent(config, run_dir)

        result = agent.run(
            query=query,
            metadata=metadata,
            run_id=run_dir.name,
        )

        save_agent_run(result, run_dir / "agent_output_trajectory.json")
        save_json(result.get("trace", []) or [], run_dir / "trace_only.json")
        save_text(str(result.get("answer", "")), run_dir / "answer.txt")

        elapsed = time.time() - start_time

        summary = summarize_result(
            query=query,
            metadata=metadata,
            result=result,
            elapsed_seconds=elapsed,
            run_dir=run_dir,
        )

        save_json(summary, run_dir / "summary.json")

        print()
        print("Status:", summary["status"])
        print("Route:", summary["route"])
        print("Domain:", summary["domain"])
        print("SQL calls:", summary["sql_calls"])
        print("API calls:", summary["api_calls"])
        print()
        print("Answer:")
        print(summary["answer"])
        print()
        print(f"Saved everything inside: {run_dir}")

        return summary

    except Exception as e:
        elapsed = time.time() - start_time

        error_payload = {
            "query": query,
            "status": "failed_exception",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 3),
            "folder": str(run_dir),
        }

        save_json(error_payload, run_dir / "error.json")

        print()
        print("FAILED:", e)
        print(f"Saved error inside: {run_dir / 'error.json'}")

        return error_payload


def run_interactive(config: CustomRunConfig) -> None:
    print()
    print("Custom query runner started.")
    print("Type your query and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        query = input("Query> ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        run_one_query(query, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one custom SQL/API agent query and save metadata, prompt, and trajectory."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom user query. If omitted, interactive mode starts.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="custom_query_runs",
        help="Base folder where each query gets a separate folder.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["ollama", "gemini"],
        help="Action-agent LLM provider.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Action-agent model name.",
    )

    parser.add_argument(
        "--answer-provider",
        type=str,
        default=ANSWER_LLM_PROVIDER,
        choices=["ollama", "gemini"],
        help="Answer-agent LLM provider.",
    )

    parser.add_argument(
        "--answer-model",
        type=str,
        default=ANSWER_MODEL,
        help="Answer-agent model name.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum action-agent steps.",
    )

    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=3,
        help="Maximum repair attempts after invalid/rejected action.",
    )

    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use real Adobe API calls instead of mock API.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CustomRunConfig(
        output_dir=args.output_dir,
        llm_provider=args.provider,
        model=args.model,
        answer_llm_provider=args.answer_provider,
        answer_model=args.answer_model,
        max_steps=args.max_steps,
        max_repair_attempts=args.max_repair_attempts,
        mock_api=not args.real_api,
    )

    print(f"Action provider: {config.llm_provider}")
    print(f"Action model: {config.model}")
    print(f"Answer provider: {config.answer_llm_provider}")
    print(f"Answer model: {config.answer_model}")
    print(f"Mock API: {config.mock_api}")

    if args.query:
        run_one_query(args.query, config)
    else:
        run_interactive(config)


if __name__ == "__main__":
    main()