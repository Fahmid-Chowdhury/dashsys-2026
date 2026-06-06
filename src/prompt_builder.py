from importlib import metadata
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from src.route_policy import (
    build_route_policy_guidance,
    get_prompt_allowed_actions,
)

MAX_EXAMPLES = 6
ROUTE_CONFIDENCE_MIN_SCORE = float(os.getenv("ROUTE_CONFIDENCE_MIN_SCORE", 0.40))


def get_selected_route(metadata: Dict[str, Any]) -> str:
    route = metadata.get("route", {})

    if isinstance(route, dict):
        return str(route.get("selected", "UNKNOWN")).upper()

    return str(route).upper()


def get_route_votes(metadata: Dict[str, Any]) -> Dict[str, float]:
    route = metadata.get("route", {})

    if not isinstance(route, dict):
        return {}

    votes = route.get("votes", {}) or {}

    result = {}
    for key, value in votes.items():
        try:
            result[str(key).upper()] = float(value)
        except Exception:
            continue

    return result


def get_route_confidence_stats(metadata: Dict[str, Any]) -> Dict[str, float]:
    selected = get_selected_route(metadata)
    votes = get_route_votes(metadata)

    if not votes or selected not in votes:
        return {
            "selected_score": 0.0,
            "second_score": 0.0,
            "margin": 0.0,
        }

    selected_score = votes[selected]
    other_scores = [
        score for route, score in votes.items()
        if route != selected
    ]

    second_score = max(other_scores) if other_scores else 0.0

    return {
        "selected_score": selected_score,
        "second_score": second_score,
        "margin": selected_score - second_score,
    }


def is_route_confident(
    metadata: Dict[str, Any],
    min_score: float = ROUTE_CONFIDENCE_MIN_SCORE,
    min_margin: float = 0.15,
) -> bool:
    stats = get_route_confidence_stats(metadata)

    return (
        stats["selected_score"] >= min_score
        and stats["margin"] >= min_margin
    )


def get_expected_tool_steps(metadata: Dict[str, Any]) -> List[str]:
    budget = metadata.get("tool_budget", {}) or {}
    expected = budget.get("expected_steps", []) or []

    return [
        step for step in expected
        if step in {"sql_query", "api_call"}
    ]


def count_completed_tool_actions(tool_results: Optional[List[Dict[str, Any]]]) -> Counter:
    counts = Counter()

    for step in tool_results or []:
        action = step.get("action")

        if action == "sql_query" and step.get("status") == "success":
            counts["sql_query"] += 1

        elif action == "api_call":
            api_call = step.get("api_call", {}) or {}
            if api_call.get("status") in {"success", "mock_success"}:
                counts["api_call"] += 1

    return counts


def get_allowed_tool_actions(metadata: Dict[str, Any]) -> List[str]:
    route = get_selected_route(metadata)
    budget = metadata.get("tool_budget", {}) or {}

    if route == "API_ONLY":
        return ["api_call"]

    if route == "SQL_ONLY":
        return ["sql_query"]

    allowed = []

    if budget.get("max_sql_calls", 0) > 0:
        allowed.append("sql_query")

    if budget.get("max_api_calls", 0) > 0:
        allowed.append("api_call")

    return allowed


def get_required_action_counts(metadata: Dict[str, Any]) -> Counter:
    route = get_selected_route(metadata)
    confident = is_route_confident(metadata)
    expected = get_expected_tool_steps(metadata)

    required = Counter()

    if route == "API_ONLY":
        required["api_call"] = 1

    elif route == "SQL_ONLY":
        required["sql_query"] = 1

    elif route in {"SQL_PLUS_API", "SQL_PLUS_API_CHAIN"} and confident:
        if expected:
            required.update(expected)
        else:
            required["sql_query"] = 1
            required["api_call"] = 1

    elif route == "API_CHAIN" and confident:
        if expected:
            required.update(expected)
        else:
            required["api_call"] = 1

    return required


def get_missing_required_actions(
    metadata: Dict[str, Any],
    tool_results: Optional[List[Dict[str, Any]]],
) -> List[str]:
    required = get_required_action_counts(metadata)
    completed = count_completed_tool_actions(tool_results)

    missing = []

    for action, needed in required.items():
        done = completed.get(action, 0)

        if done < needed:
            remaining = needed - done
            missing.append(action if remaining == 1 else f"{action} x{remaining}")

    return missing


# def build_route_policy_guidance(
#     metadata: Dict[str, Any],
#     tool_results: Optional[List[Dict[str, Any]]],
# ) -> str:
#     route = get_selected_route(metadata)
#     stats = get_route_confidence_stats(metadata)
#     confident = is_route_confident(metadata)

#     allowed_tools = get_allowed_tool_actions(metadata)
#     required = get_required_action_counts(metadata)
#     missing = get_missing_required_actions(metadata, tool_results)
#     completed = count_completed_tool_actions(tool_results)

#     lines = []

#     lines.append(f"Selected route: {route}")
#     lines.append(
#         f"Route confidence: selected_score={stats['selected_score']:.4f}, "
#         f"margin={stats['margin']:.4f}, confident={confident}"
#     )
#     lines.append(f"Allowed tool actions: {allowed_tools}")

#     if required:
#         lines.append(f"Required tool evidence before final_answer: {dict(required)}")

#         if missing:
#             lines.append(f"Missing required action(s): {missing}")
#             lines.append(
#                 "You may choose any missing required tool action next. "
#                 "Do not return final_answer yet."
#             )
#         else:
#             lines.append(
#                 "All required tool evidence is collected. "
#                 "Return final_answer using only verified tool results."
#             )

#     else:
#         lines.append(
#             "Route is uncertain or flexible. Choose the most useful allowed tool action. "
#             "Do not return final_answer until at least one verified tool result is available."
#         )
#         lines.append(f"Completed tool actions so far: {dict(completed)}")

#     return "\n".join(lines)


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_columns(columns: List[Dict[str, Any]], max_columns: int = 25) -> List[str]:
    result = []

    for col in columns[:max_columns]:
        name = col.get("name")
        col_type = col.get("type")

        if name and col_type:
            result.append(f"{name} ({col_type})")
        elif name:
            result.append(name)

    return result


def format_allowed_tables(metadata: Dict[str, Any]) -> str:
    allowed_tables = metadata.get("allowed_tables", {}) or {}

    if not allowed_tables:
        return "No SQL tables are allowed for this query."

    lines = []

    for table_name, table_info in allowed_tables.items():
        lines.append(f"- Table: {table_name}")
        lines.append(f"  Domain: {table_info.get('domain', 'unknown')}")
        lines.append(f"  Kind: {table_info.get('kind', 'unknown')}")
        lines.append(f"  ID columns: {table_info.get('id_columns', [])}")

        columns = compact_columns(table_info.get("columns", []))
        lines.append("  Columns:")
        for col in columns:
            lines.append(f"    - {col}")

    return "\n".join(lines)


def format_join_candidates(metadata: Dict[str, Any]) -> str:
    joins = metadata.get("join_candidates", []) or []

    if not joins:
        return "No joins are required or allowed unless clearly necessary."

    lines = []

    for join in joins:
        left_table = join.get("left_table")
        left_column = join.get("left_column")
        right_table = join.get("right_table")
        right_column = join.get("right_column")
        confidence = join.get("confidence", "unknown")

        lines.append(
            f"- {left_table}.{left_column} = {right_table}.{right_column} "
            f"(confidence: {confidence})"
        )

    return "\n".join(lines)


def format_allowed_api_endpoints(metadata: Dict[str, Any]) -> str:
    endpoints = metadata.get("allowed_api_endpoints", []) or []

    if not endpoints:
        return "No API endpoints are allowed for this query."

    lines = []

    for endpoint in endpoints:
        lines.append(f"- Endpoint: {endpoint.get('endpoint')}")
        lines.append(f"  Method: {endpoint.get('method')}")
        lines.append(f"  Path: {endpoint.get('path')}")
        lines.append(f"  Domain: {endpoint.get('domain', 'unknown')}")

        param_names = endpoint.get("param_names", []) or []
        if param_names:
            lines.append(f"  Allowed query params: {param_names}")

        observed_params = endpoint.get("observed_params", {}) or {}
        if observed_params:
            lines.append("  Observed param examples:")
            for key, values in observed_params.items():
                lines.append(f"    - {key}: {values[:5]}")

        openapi = endpoint.get("openapi")
        if openapi:
            query_params = openapi.get("query_param_names", []) or []
            path_params = openapi.get("path_param_names", []) or []
            required_params = openapi.get("required_params", []) or []

            if query_params:
                lines.append(f"  OpenAPI query params: {query_params}")

            if path_params:
                lines.append(f"  OpenAPI path params: {path_params}")

            if required_params:
                lines.append(f"  Required params: {required_params}")

    return "\n".join(lines)


def format_similar_examples(metadata: Dict[str, Any], max_examples: int = MAX_EXAMPLES) -> str:
    examples = metadata.get("similar_examples", []) or []

    if not examples:
        return "No similar examples available."

    lines = []

    for idx, ex in enumerate(examples[:max_examples], start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"- Query: {ex.get('query')}")
        lines.append(f"- Route: {ex.get('route')}")
        lines.append(f"- Domain: {ex.get('domain')}")
        lines.append(f"- Intent: {ex.get('intent')}")
        lines.append(f"- SQL tables: {ex.get('sql_tables', [])}")
        lines.append(f"- API endpoints: {ex.get('api_endpoints', [])}")
        
        if ex.get("retrieval_sources"):
            lines.append(f"- Retrieval sources: {ex.get('retrieval_sources')}")

        if ex.get("matched_subqueries"):
            lines.append(f"- Matched subqueries: {ex.get('matched_subqueries')}")

        gold_sql = ex.get("gold_sql")
        if gold_sql:
            lines.append("- Example SQL:")
            lines.append(gold_sql.strip())

        gold_api = ex.get("gold_api", [])
        if gold_api:
            lines.append(f"- Example API: {gold_api}")

        lines.append("")

    return "\n".join(lines).strip()


def format_tool_results(tool_results: Optional[List[Dict[str, Any]]]) -> str:
    if not tool_results:
        return "No tool results yet."

    lines = []

    for result in tool_results:
        step = result.get("step")
        action = result.get("action")
        status = result.get("status", "unknown")

        lines.append(f"Step {step}: {action}")
        lines.append(f"Status: {status}")

        if action == "sql_query":
            lines.append("SQL:")
            lines.append(str(result.get("sql", "")).strip())
            lines.append("Result preview:")
            lines.append(json.dumps(result.get("results", []), indent=2, ensure_ascii=False))

        elif action == "api_call":
            api_call = result.get("api_call", {})
            lines.append("API call:")
            lines.append(json.dumps(api_call, indent=2, ensure_ascii=False))
            lines.append("Result preview:")
            lines.append(json.dumps(api_call.get("result_preview", []), indent=2, ensure_ascii=False))

        if result.get("error"):
            lines.append(f"Error: {result.get('error')}")

        lines.append("")

    return "\n".join(lines).strip()


def build_output_instruction(
    metadata: Dict[str, Any],
    tool_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    allowed_actions = get_prompt_allowed_actions(
        metadata=metadata,
        trace=tool_results,
    )

    return f"""
You must return ONLY one valid JSON object.

Allowed next actions right now:
{allowed_actions}

If the allowed next action is sql_query, return:
{{
  "action": "sql_query",
  "sql": "SELECT ..."
}}

If the allowed next action is api_call, return:
{{
  "action": "api_call",
  "method": "GET",
  "url": "/path",
  "params": {{}}
}}

If the allowed next action is final_answer, return:
{{
  "action": "final_answer",
  "answer": "..."
}}

Rules:
- Return only JSON.
- Do not use markdown.
- Do not explain outside JSON.
- Do not call tools not allowed by the route.
- Do not exceed the tool budget.
- Do not repeat any SQL/API call already shown in CURRENT VERIFIED TOOL RESULTS.
- If a previous SQL returned empty rows, do not retry the exact same SQL.
- If a previous API call returned empty/error, do not retry the exact same API call.
- A second SQL/API call is allowed only if it is meaningfully different.
- Do not invent table names.
- Do not invent column names.
- Do not invent API endpoints.
- Do not leave placeholders like {{schema_id}} or <destination_id>.
- If required evidence is missing, call one of the allowed tool actions first.
""".strip()


def build_agent_system_prompt() -> str:
    return """
You are a metadata-grounded SQL/API planning agent.

Your job is to answer user queries by choosing the next valid action.

You have access to two tools:
1. sql_query: runs read-only SQL over the local knowledge graph snapshot.
2. api_call: calls an Adobe API endpoint.

You must follow the provided metadata strictly.

Important behavior:
- Use only allowed tables.
- Use only allowed columns.
- Use only allowed API endpoints.
- Use only allowed query parameters.
- Use similar examples only as guidance, not as facts.
- Use verified tool results as the only evidence for the final answer.
- If SQL and API results disagree, mention the discrepancy in the final answer.
- Prefer the minimum number of tool calls.
- Never repeat the exact same SQL query or API call.
- If a tool result is empty, either change strategy, use another allowed tool, or return final_answer if enough evidence exists.
- Tool budget is dynamic: every executed SQL/API call consumes one budget slot.
- For list questions, return concise items with relevant IDs.
- For count questions, return the count clearly.
- For status/time questions, return the exact status/time if available.
- User-requested output names may not match actual SQL column names.
- Always use the exact column names shown in ALLOWED SQL TABLES.
- You may alias real columns to user-friendly names, but never select non-existent columns.
- Example: if the user asks for audienceId but the table has segmentid, select segmentid AS audienceId.
- Never call an API endpoint containing {placeholder} or <placeholder> unless you can fully replace it with a verified value from SQL/API results.
- If required evidence is already collected and no safe API call exists, return final_answer from verified SQL results.
""".strip()


def build_agent_user_prompt(
    query: str,
    metadata: Dict[str, Any],
    tool_results: Optional[List[Dict[str, Any]]] = None,
    verifier_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    route = metadata.get("route", {}).get("selected", "unknown")
    domain = metadata.get("domain", {}).get("selected", "unknown")
    intent = metadata.get("intent", "unknown")
    budget = metadata.get("tool_budget", {}) or {}
    constraints = metadata.get("answer_constraints", []) or []

    sections = []

    sections.append("USER QUERY")
    sections.append(query.strip())

    sections.append("\nSELECTED METADATA")
    sections.append(f"Route: {route}")
    sections.append(f"Domain: {domain}")
    sections.append(f"Intent: {intent}")
    sections.append(f"Tool budget: {json.dumps(budget, ensure_ascii=False)}")
    
    sections.append("\nQUERY ANALYSIS")
    sections.append(format_query_analysis(metadata))
    
    sections.append("\nROUTE-AWARE TOOL POLICY")
    sections.append(
        build_route_policy_guidance(
            metadata=metadata,
            trace=tool_results,
        )
    )

    sections.append("\nALLOWED SQL TABLES")
    sections.append(format_allowed_tables(metadata))

    sections.append("\nALLOWED JOIN CANDIDATES")
    sections.append(format_join_candidates(metadata))

    sections.append("\nALLOWED API ENDPOINTS")
    sections.append(format_allowed_api_endpoints(metadata))

    sections.append("\nSIMILAR EXAMPLES")
    sections.append(format_similar_examples(metadata))

    sections.append("\nCURRENT VERIFIED TOOL RESULTS")
    sections.append(format_tool_results(tool_results))

    if verifier_feedback:
        sections.append("\nVERIFIER FEEDBACK FROM PREVIOUS INVALID ACTION")
        sections.append(json.dumps(verifier_feedback, indent=2, ensure_ascii=False))

    if constraints:
        sections.append("\nANSWER CONSTRAINTS")
        for c in constraints:
            sections.append(f"- {c}")

    sections.append("\nOUTPUT FORMAT")
    sections.append(
        build_output_instruction(
            metadata=metadata,
            tool_results=tool_results,
        )
    )

    return "\n".join(sections)


def format_query_analysis(metadata: Dict[str, Any]) -> str:
    analysis = metadata.get("query_analysis", {}) or {}

    if not analysis:
        return "No query analysis available."

    lines = []

    lines.append(f"Complexity: {analysis.get('complexity', 'unknown')}")
    lines.append(f"Confidence: {analysis.get('confidence', 'unknown')}")
    lines.append(f"Reason: {analysis.get('reason', '')}")

    subqueries = analysis.get("subqueries", []) or []

    if subqueries:
        lines.append("Subqueries:")

        for idx, item in enumerate(subqueries, start=1):
            lines.append(
                f"{idx}. [{item.get('purpose', 'other')}] {item.get('query')}"
            )

    return "\n".join(lines)


def build_prompt_messages(
    query: str,
    metadata: Dict[str, Any],
    tool_results: Optional[List[Dict[str, Any]]] = None,
    verifier_feedback: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_agent_system_prompt(),
        },
        {
            "role": "user",
            "content": build_agent_user_prompt(
                query=query,
                metadata=metadata,
                tool_results=tool_results,
                verifier_feedback=verifier_feedback,
            ),
        },
    ]


def save_prompt_messages(messages: List[Dict[str, str]], output_path: str | Path) -> None:
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def print_prompt_preview(messages: List[Dict[str, str]], max_chars: int = 3000) -> None:
    print("Prompt preview")
    print("==============")
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        content = msg["content"]

        if len(content) > max_chars:
            print(content[:max_chars])
            print("\n... [truncated preview]")
        else:
            print(content)


if __name__ == "__main__":
    metadata = load_json("data/sample_metadata.json")
    query = metadata.get("query", "List all journeys")

    messages = build_prompt_messages(
        query=query,
        metadata=metadata,
        tool_results=None,
        verifier_feedback=None,
    )

    save_prompt_messages(messages, "data/sample_prompt_messages.json")
    print_prompt_preview(messages)
    print("\nSaved to: data/sample_prompt_messages.json")