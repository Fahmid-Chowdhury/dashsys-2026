import json
from typing import Any, Dict, List, Tuple

from src.answer_builder import (
    extract_api_steps,
    extract_sql_steps,
    get_sql_rows,
    summarize_api_step,
)


ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The final user-facing answer written only from verified SQL/API evidence.",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


ANSWER_SYSTEM_PROMPT = """
You are the answer agent for a SQL/API data agent.

Your only job:
- Write the final user-facing answer.

Strict rules:
- Do NOT generate SQL.
- Do NOT generate API calls.
- Do NOT suggest another tool call.
- Do NOT modify the trace.
- Do NOT change the output format of the outer agent.
- Use ONLY the verified SQL/API evidence provided in the evidence object.
- Do not invent IDs, names, counts, dates, statuses, fields, or relationships.
- If SQL evidence and API evidence disagree, mention the discrepancy clearly.
- If a SQL/API call failed, mention the failure briefly only if it affects the answer.
- If evidence is incomplete, say what could be answered and what could not be verified.
- Preserve the field names requested by the user when possible.
- Keep the answer concise, direct, and useful.

Return ONLY valid JSON in this exact shape:
{
  "answer": "..."
}
""".strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in answer-agent output.")

    parsed = json.loads(text[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Answer-agent output must be a JSON object.")

    return parsed


def compact_sql_steps(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []

    for step in extract_sql_steps(trace):
        rows = step.get("results", []) or []

        compact.append(
            {
                "step": step.get("step"),
                "action": "sql_query",
                "sql": step.get("sql"),
                "status": step.get("status"),
                "rows": rows,
                "row_count_in_preview": len(rows),
                "error": step.get("error"),
                "warnings": step.get("warnings", []),
            }
        )

    return compact


def compact_api_steps(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []

    for step in extract_api_steps(trace):
        api_call = step.get("api_call", {}) or {}
        summary = summarize_api_step(step)

        compact.append(
            {
                "step": step.get("step"),
                "action": "api_call",
                "method": api_call.get("method"),
                "url": api_call.get("url"),
                "params": api_call.get("params", {}),
                "status": api_call.get("status"),
                "status_code": api_call.get("status_code"),
                "total_count": summary.get("total_count"),
                "item_count": summary.get("item_count"),
                "items": summary.get("items", []),
                "error": api_call.get("error"),
            }
        )

    return compact


def build_answer_evidence(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evidence object for the answer agent.

    Important:
    - This does not include similar examples as final evidence.
    - This does not include the action agent's draft answer.
    - The answer agent sees verified trace evidence only.
    """

    return {
        "query": query,
        "route": metadata.get("route"),
        "intent": metadata.get("intent"),
        "domain": metadata.get("domain"),
        "sql_rows_combined": get_sql_rows(trace),
        "sql_steps": compact_sql_steps(trace),
        "api_steps": compact_api_steps(trace),
    }


def build_answer_messages(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    evidence = build_answer_evidence(
        query=query,
        metadata=metadata,
        trace=trace,
    )

    return [
        {
            "role": "system",
            "content": ANSWER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(evidence, indent=2, ensure_ascii=False, default=str),
        },
    ]


def generate_answer_with_llm(
    llm_client: Any,
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    messages = build_answer_messages(
        query=query,
        metadata=metadata,
        trace=trace,
    )

    raw_output = llm_client.generate(messages)
    parsed = safe_json_loads(raw_output)

    answer = str(parsed.get("answer", "")).strip()

    if not answer:
        answer = "No verified SQL/API evidence was available to answer the query."

    debug = {
        "event": "answer_agent_output",
        "raw_output": raw_output,
        "parsed": parsed,
    }

    return answer, debug