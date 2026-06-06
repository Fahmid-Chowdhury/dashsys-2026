from collections import Counter
import os
from typing import Any, Dict, List, Optional

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
    result: Dict[str, float] = {}

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
        score
        for route, score in votes.items()
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
        step
        for step in expected
        if step in {"sql_query", "api_call"}
    ]


def get_allowed_tool_actions(metadata: Dict[str, Any]) -> List[str]:
    route = get_selected_route(metadata)
    budget = metadata.get("tool_budget", {}) or {}

    if route == "API_ONLY":
        return ["api_call"]

    if route == "SQL_ONLY":
        return ["sql_query"]

    allowed: List[str] = []

    if budget.get("max_sql_calls", 0) > 0:
        allowed.append("sql_query")

    if budget.get("max_api_calls", 0) > 0:
        allowed.append("api_call")

    return allowed


def count_completed_tool_actions(trace: Optional[List[Dict[str, Any]]]) -> Counter:
    counts = Counter()

    for step in trace or []:
        action = step.get("action")

        if action == "sql_query":
            if step.get("status") == "success":
                counts["sql_query"] += 1

        elif action == "api_call":
            api_call = step.get("api_call", {}) or {}

            if api_call.get("status") in {"success", "mock_success"}:
                counts["api_call"] += 1

    return counts


def count_executed_tool_actions(trace: Optional[List[Dict[str, Any]]]) -> Counter:
    """
    Counts executed tool calls, regardless of whether the tool returned rows.

    This is used for budget consumption.
    If a SQL/API call was executed, it consumed budget.
    """

    counts = Counter()

    for step in trace or []:
        action = step.get("action")

        if action == "sql_query":
            counts["sql_query"] += 1

        elif action == "api_call":
            counts["api_call"] += 1

    return counts


def get_required_action_counts(
    metadata: Dict[str, Any],
    route_confident: bool,
) -> Counter:
    """
    Defines minimum route coverage.

    This does NOT force a fixed sequence like:
        SQL first, then API.

    It only says:
    - SQL_ONLY needs SQL coverage.
    - API_ONLY/API_CHAIN needs API coverage.
    - SQL_PLUS_API / SQL_PLUS_API_CHAIN need both SQL and API coverage
      when the route is confident and the budget allows them.

    Coverage means the agent must at least attempt that tool type.
    Exact number of calls is still controlled by the dynamic tool budget.
    """

    route = get_selected_route(metadata)
    budget = metadata.get("tool_budget", {}) or {}

    max_sql = int(budget.get("max_sql_calls", 0) or 0)
    max_api = int(budget.get("max_api_calls", 0) or 0)

    expected_steps = get_expected_tool_steps(metadata)

    required = Counter()

    if route == "SQL_ONLY":
        if max_sql > 0:
            required["sql_query"] = 1
        return required

    if route in {"API_ONLY", "API_CHAIN"}:
        if max_api > 0:
            required["api_call"] = 1
        return required

    if route in {"SQL_PLUS_API", "SQL_PLUS_API_CHAIN"}:
        if not route_confident:
            return required

        # Prefer metadata expected_steps when available.
        # But cap by available budget so the policy does not demand impossible calls.
        if expected_steps:
            expected_counts = Counter(expected_steps)

            if max_sql > 0 and expected_counts.get("sql_query", 0) > 0:
                required["sql_query"] = min(expected_counts["sql_query"], max_sql)

            if max_api > 0 and expected_counts.get("api_call", 0) > 0:
                required["api_call"] = min(expected_counts["api_call"], max_api)

            return required

        # Fallback: mixed route requires both tool types, if budget allows.
        if max_sql > 0:
            required["sql_query"] = 1

        if max_api > 0:
            required["api_call"] = 1

        return required

    return required


def get_missing_required_actions(
    required: Counter,
    completed: Counter,
) -> List[str]:
    missing: List[str] = []

    for action, needed_count in required.items():
        done_count = completed.get(action, 0)

        if done_count < needed_count:
            remaining = needed_count - done_count

            if remaining == 1:
                missing.append(action)
            else:
                missing.append(f"{action} x{remaining}")

    return missing


def get_missing_required_base_actions(
    required: Counter,
    completed: Counter,
) -> List[str]:
    missing: List[str] = []

    for action, needed_count in required.items():
        done_count = completed.get(action, 0)

        if done_count < needed_count:
            missing.append(action)

    return missing


def get_remaining_tool_actions(
    metadata: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]],
) -> List[str]:
    allowed = get_allowed_tool_actions(metadata)
    executed = count_executed_tool_actions(trace)
    budget = metadata.get("tool_budget", {}) or {}

    remaining: List[str] = []

    if "sql_query" in allowed:
        if executed.get("sql_query", 0) < budget.get("max_sql_calls", 0):
            remaining.append("sql_query")

    if "api_call" in allowed:
        if executed.get("api_call", 0) < budget.get("max_api_calls", 0):
            remaining.append("api_call")

    return remaining


def get_remaining_tool_budget(
    metadata: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]],
) -> Dict[str, int]:
    budget = metadata.get("tool_budget", {}) or {}
    executed = count_executed_tool_actions(trace)

    max_sql = int(budget.get("max_sql_calls", 0) or 0)
    max_api = int(budget.get("max_api_calls", 0) or 0)

    used_sql = int(executed.get("sql_query", 0))
    used_api = int(executed.get("api_call", 0))

    return {
        "max_sql_calls": max_sql,
        "used_sql_calls": used_sql,
        "remaining_sql_calls": max(0, max_sql - used_sql),
        "max_api_calls": max_api,
        "used_api_calls": used_api,
        "remaining_api_calls": max(0, max_api - used_api),
    }


def get_prompt_allowed_actions(
    metadata: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]],
    min_score: float = ROUTE_CONFIDENCE_MIN_SCORE,
    min_margin: float = 0.15,
    min_tools_before_final_when_uncertain: int = 1,
) -> List[str]:
    route_confident = is_route_confident(
        metadata=metadata,
        min_score=min_score,
        min_margin=min_margin,
    )

    executed = count_executed_tool_actions(trace)
    completed = count_completed_tool_actions(trace)
    remaining_tools = get_remaining_tool_actions(metadata, trace)

    required = get_required_action_counts(
        metadata=metadata,
        route_confident=route_confident,
    )

    if required:
        # Use executed counts for route coverage so a failed API attempt still consumes budget
        # and does not trap the agent forever.
        missing_base = get_missing_required_base_actions(
            required=required,
            completed=executed,
        )

        missing_with_budget = [
            action for action in missing_base
            if action in remaining_tools
        ]

        if missing_with_budget:
            return missing_with_budget

        # Required tool type was attempted, or no budget remains for it.
        # The model may continue with optional remaining tools or finish.
        return remaining_tools + ["final_answer"]

    total_completed = sum(completed.values())

    if total_completed < min_tools_before_final_when_uncertain:
        return remaining_tools

    return remaining_tools + ["final_answer"]


def check_route_aware_policy(
    action: Dict[str, Any],
    metadata: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]],
    min_score: float = ROUTE_CONFIDENCE_MIN_SCORE,
    min_margin: float = 0.15,
    min_tools_before_final_when_uncertain: int = 1,
) -> Optional[Dict[str, Any]]:
    action_name = action.get("action")

    if action_name not in {"sql_query", "api_call", "final_answer"}:
        return {
            "ok": False,
            "target": "route_policy",
            "errors": [f"Unknown action: {action_name}"],
            "repair_hint": "Return sql_query, api_call, or final_answer.",
        }

    route = get_selected_route(metadata)

    route_confident = is_route_confident(
        metadata=metadata,
        min_score=min_score,
        min_margin=min_margin,
    )

    allowed_tools = get_allowed_tool_actions(metadata)
    completed = count_completed_tool_actions(trace)

    required = get_required_action_counts(
        metadata=metadata,
        route_confident=route_confident,
    )

    executed = count_executed_tool_actions(trace)

    missing_required = get_missing_required_actions(
        required=required,
        completed=executed,
    )

    if action_name in {"sql_query", "api_call"}:
        if action_name not in allowed_tools:
            return {
                "ok": False,
                "target": "route_policy",
                "errors": [
                    f"Action {action_name} is not allowed for route {route}.",
                    f"Allowed tool actions: {allowed_tools}",
                ],
                "repair_hint": (
                    f"Use only the allowed tool action(s): {allowed_tools}. "
                    "If enough evidence is already available, return final_answer."
                ),
            }

        budget = metadata.get("tool_budget", {}) or {}
        executed = count_executed_tool_actions(trace)

        if action_name == "sql_query":
            max_allowed = budget.get("max_sql_calls", 0)
        else:
            max_allowed = budget.get("max_api_calls", 0)

        if executed.get(action_name, 0) >= max_allowed:
            return {
                "ok": False,
                "target": "route_policy",
                "errors": [
                    f"Tool budget exhausted for {action_name}.",
                    f"Executed {executed.get(action_name, 0)} out of allowed {max_allowed}.",
                ],
                "repair_hint": (
                    "Use another allowed tool action if needed, "
                    "or return final_answer if enough evidence is available."
                ),
            }

        if required:
            missing_base = get_missing_required_base_actions(
                required=required,
                completed=completed,
            )

            if missing_base and action_name not in missing_base:
                return {
                    "ok": False,
                    "target": "route_policy",
                    "errors": [
                        f"Required evidence is still missing: {missing_base}.",
                        f"Proposed action {action_name} is optional right now and should wait.",
                    ],
                    "repair_hint": (
                        f"Use one of the missing required action(s): {missing_base}."
                    ),
                }

        return None

    if action_name == "final_answer":
        if required and missing_required:
            remaining_tools = get_remaining_tool_actions(metadata, trace)

            blocking_missing = [
                action for action in get_missing_required_base_actions(required, executed)
                if action in remaining_tools
            ]

            if blocking_missing:
                return {
                    "ok": False,
                    "target": "route_policy",
                    "errors": [
                        f"Cannot return final_answer yet. Missing required tool action(s): {blocking_missing}."
                    ],
                    "repair_hint": (
                        f"Complete one of the missing required action(s): {blocking_missing}. "
                        "Then return final_answer."
                    ),
                }

        if not required and allowed_tools:
            total_completed = sum(completed.values())

            if total_completed < min_tools_before_final_when_uncertain:
                return {
                    "ok": False,
                    "target": "route_policy",
                    "errors": [
                        "Route is uncertain, but no verified tool evidence has been collected yet."
                    ],
                    "repair_hint": (
                        f"Choose one useful tool action from {allowed_tools}. "
                        "Use verified results before returning final_answer."
                    ),
                }

        return None

    return None


def build_route_policy_guidance(
    metadata: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]],
    min_score: float = ROUTE_CONFIDENCE_MIN_SCORE,
    min_margin: float = 0.15,
    min_tools_before_final_when_uncertain: int = 1,
) -> str:
    route = get_selected_route(metadata)

    stats = get_route_confidence_stats(metadata)

    confident = is_route_confident(
        metadata=metadata,
        min_score=min_score,
        min_margin=min_margin,
    )

    allowed_tools = get_allowed_tool_actions(metadata)
    completed = count_completed_tool_actions(trace)
    executed = count_executed_tool_actions(trace)
    remaining_budget = get_remaining_tool_budget(metadata, trace)

    required = get_required_action_counts(
        metadata=metadata,
        route_confident=confident,
    )

    missing = get_missing_required_actions(
        required=required,
        completed=completed,
    )

    prompt_allowed = get_prompt_allowed_actions(
        metadata=metadata,
        trace=trace,
        min_score=min_score,
        min_margin=min_margin,
        min_tools_before_final_when_uncertain=min_tools_before_final_when_uncertain,
    )

    lines: List[str] = []

    lines.append(f"Selected route: {route}")
    lines.append(
        f"Route confidence: selected_score={stats['selected_score']:.4f}, "
        f"second_score={stats['second_score']:.4f}, "
        f"margin={stats['margin']:.4f}, confident={confident}"
    )
    lines.append(f"Executed tool actions: {dict(executed)}")
    lines.append(f"Completed successful tool actions: {dict(completed)}")
    lines.append(f"Remaining tool budget: {remaining_budget}")
    lines.append(f"Allowed tool actions: {allowed_tools}")
    lines.append(f"Allowed next actions now: {prompt_allowed}")

    if required:
        lines.append(f"Required evidence before final_answer: {dict(required)}")

        if missing:
            lines.append(f"Missing required evidence: {missing}")
            lines.append(
                "You may choose any missing required tool action next. "
                "Do not return final_answer yet."
            )
        else:
            lines.append(
                "All required evidence has been collected. "
                "Return final_answer using only verified tool results."
            )
    else:
        lines.append(
            "Route is flexible. Choose the most useful allowed tool action within the remaining budget. "
            "Do not repeat an already executed SQL/API call. "
            "If enough evidence is available, return final_answer."
        )

    return "\n".join(lines)