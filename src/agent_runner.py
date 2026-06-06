import json
from collections import Counter
from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.route_policy import check_route_aware_policy as route_policy_check
from src.prompt_builder import build_prompt_messages
from src.answer_builder import build_final_answer
from src.tool_executor import ToolExecutor
from src.verifier import (
    load_api_index_if_available,
    validate_tool_call,
)

ROUTE_CONFIDENCE_MIN_SCORE = float(os.getenv("ROUTE_CONFIDENCE_MIN_SCORE", 0.40))

@dataclass
class AgentRunConfig:
    max_steps: int = 5
    max_repair_attempts: int = 2
    snapshot_dir: str = "DBSnapshot"
    api_index_path: str = "data/api_index_enriched.json"
    mock_api: bool = True
    save_prompts: bool = True
    prompt_output_dir: str = "data/agent_prompts"
    use_answer_builder: bool = True
    
    route_confidence_min_score: float = ROUTE_CONFIDENCE_MIN_SCORE
    route_confidence_min_margin: float = 0.15
    min_tools_before_final_when_uncertain: int = 1


class BaseLLMClient:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


ACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["sql_query", "api_call", "final_answer"],
        },
        "sql": {
            "type": "string",
            "description": "Read-only SQL query when action is sql_query.",
        },
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
        },
        "url": {
            "type": "string",
            "description": "API path or URL when action is api_call.",
        },
        "params": {
            "type": "object",
            "description": "API query parameters.",
            "additionalProperties": True,
        },
        "headers": {
            "type": "object",
            "description": "Optional API headers.",
            "additionalProperties": True,
        },
        "body": {
            "type": ["object", "null"],
            "description": "Optional API request body.",
            "additionalProperties": True,
        },
        "answer": {
            "type": "string",
            "description": "Final answer when action is final_answer.",
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


class OllamaLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        num_ctx: int = 32768,
        use_json_schema: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.use_json_schema = use_json_schema

    def generate(self, messages: List[Dict[str, str]]) -> str:
        import ollama

        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
            "stream": False,
        }

        if self.use_json_schema:
            kwargs["format"] = ACTION_JSON_SCHEMA
        else:
            kwargs["format"] = "json"

        response = ollama.chat(**kwargs)

        # Supports both dict-style and object-style responses.
        if isinstance(response, dict):
            return response["message"]["content"]

        return response.message.content
    

class GeminiLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        use_json_schema: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.use_json_schema = use_json_schema

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []

        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            parts.append(f"{role}:\n{content}")

        parts.append(
            "Return only valid JSON. Do not include markdown, comments, or explanations."
        )

        return "\n\n".join(parts)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        from google import genai

        client = genai.Client()
        prompt = self._messages_to_prompt(messages)

        config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }

        if self.use_json_schema:
            config["response_json_schema"] = ACTION_JSON_SCHEMA

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,  # type: ignore[arg-type]
        )

        return response.text or ""


class StaticLLMClient(BaseLLMClient):
    """
    Deterministic testing client.

    It returns predefined JSON actions one by one.
    """

    def __init__(self, actions: List[Dict[str, Any]]):
        self.actions = actions
        self.index = 0

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.index >= len(self.actions):
            return json.dumps(
                {
                    "action": "final_answer",
                    "answer": "No more predefined actions.",
                }
            )

        action = self.actions[self.index]
        self.index += 1
        return json.dumps(action, ensure_ascii=False)
    
    
def create_llm_client(
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> BaseLLMClient:
    provider = provider.lower().strip()

    if provider == "ollama":
        return OllamaLLMClient(
            model=model,
            temperature=temperature,
            num_ctx=32768,
            use_json_schema=True,
        )

    if provider == "gemini":
        return GeminiLLMClient(
            model=model,
            temperature=temperature,
            use_json_schema=True,
        )

    raise ValueError(
        f"Unknown LLM provider: {provider}. Use 'ollama' or 'gemini'."
    )


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_json_action(text: str) -> Dict[str, Any]:
    """
    Parses the LLM output.

    The prompt tells the LLM to return only JSON.
    This function still tries to recover if the model adds extra text.
    """

    text = text.strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        raise ValueError("JSON output must be an object.")

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")

    candidate = text[start:end + 1]
    parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Recovered JSON is not an object.")

    return parsed

def normalize_param_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1:
            return normalize_param_value(value[0])

        return ",".join(str(v) for v in value)

    if isinstance(value, dict):
        return {
            str(k): normalize_param_value(v)
            for k, v in value.items()
        }

    return value


def normalize_api_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(k): normalize_param_value(v)
        for k, v in (params or {}).items()
    }


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


def count_successful_tool_actions(trace: List[Dict[str, Any]]) -> Counter:
    counts = Counter()

    for step in trace:
        action = step.get("action")

        if action == "sql_query" and step.get("status") == "success":
            counts["sql_query"] += 1

        elif action == "api_call":
            api_call = step.get("api_call", {}) or {}
            if api_call.get("status") in {"success", "mock_success"}:
                counts["api_call"] += 1

    return counts


def get_required_action_counts(
    metadata: Dict[str, Any],
    route_confident: bool,
) -> Counter:
    route = get_selected_route(metadata)
    expected_steps = get_expected_tool_steps(metadata)

    required = Counter()

    if route == "API_ONLY":
        required["api_call"] = 1
        return required

    if route == "SQL_ONLY":
        required["sql_query"] = 1
        return required

    if route in {"SQL_PLUS_API", "SQL_PLUS_API_CHAIN"} and route_confident:
        if expected_steps:
            required.update(expected_steps)
        else:
            required["sql_query"] = 1
            required["api_call"] = 1

        return required

    if route == "API_CHAIN" and route_confident:
        if expected_steps:
            required.update(expected_steps)
        else:
            required["api_call"] = 1

        return required

    return required


def get_missing_required_actions(
    required: Counter,
    completed: Counter,
) -> List[str]:
    missing = []

    for action, needed_count in required.items():
        done_count = completed.get(action, 0)

        if done_count < needed_count:
            remaining = needed_count - done_count

            if remaining == 1:
                missing.append(action)
            else:
                missing.append(f"{action} x{remaining}")

    return missing


def normalize_api_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allows both formats:

    {
      "action": "api_call",
      "method": "GET",
      "url": "/x",
      "params": {}
    }

    and:

    {
      "action": "api_call",
      "api_call": {
        "method": "GET",
        "url": "/x",
        "params": {}
      }
    }
    """

    if action.get("action") != "api_call":
        return action

    if "api_call" in action and isinstance(action["api_call"], dict):
        api_call = action["api_call"]

        return {
            "action": "api_call",
            "method": api_call.get("method", "GET"),
            "url": api_call.get("url", ""),
            "params": normalize_api_params(api_call.get("params", {}) or {}),
            "headers": api_call.get("headers", {}) or {},
            "body": api_call.get("body"),
        }

    return {
        "action": "api_call",
        "method": action.get("method", "GET"),
        "url": action.get("url", ""),
        "params": normalize_api_params(action.get("params", {}) or {}),
        "headers": action.get("headers", {}) or {},
        "body": action.get("body"),
    }


def count_trace_actions(trace: List[Dict[str, Any]], action_name: str) -> int:
    return sum(1 for step in trace if step.get("action") == action_name)


def normalize_sql_for_signature(sql: str) -> str:
    """
    Normalizes SQL so exact duplicate queries are easier to detect.

    This does not try to detect semantic duplicates.
    It only catches the same SQL with whitespace/case differences.
    """

    sql = str(sql or "").strip()
    sql = sql.rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def stable_json_dumps(value: Any) -> str:
    """
    Stable JSON string for comparing params/body safely.
    """

    return json.dumps(
        value or {},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def build_tool_call_signature(action: Dict[str, Any]) -> Optional[str]:
    """
    Creates a stable signature for SQL/API actions.

    Same SQL => same signature.
    Same API method + URL + params + body => same signature.
    """

    action_name = action.get("action")

    if action_name == "sql_query":
        return f"sql_query::{normalize_sql_for_signature(action.get('sql', ''))}"

    if action_name == "api_call":
        method = str(action.get("method", "GET")).upper().strip()
        url = str(action.get("url", "")).strip()
        params = stable_json_dumps(action.get("params", {}) or {})
        body = stable_json_dumps(action.get("body", {}) or {})

        return f"api_call::{method}::{url}::params={params}::body={body}"

    return None


def build_executed_step_signature(step: Dict[str, Any]) -> Optional[str]:
    """
    Builds a signature from a previously executed trace step.
    """

    action_name = step.get("action")

    if action_name == "sql_query":
        return f"sql_query::{normalize_sql_for_signature(step.get('sql', ''))}"

    if action_name == "api_call":
        api_call = step.get("api_call", {}) or {}

        method = str(api_call.get("method", "GET")).upper().strip()
        url = str(api_call.get("url", "")).strip()
        params = stable_json_dumps(api_call.get("params", {}) or {})
        body = stable_json_dumps(api_call.get("body", {}) or {})

        return f"api_call::{method}::{url}::params={params}::body={body}"

    return None


def check_duplicate_tool_call(
    action: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Rejects an exact duplicate SQL/API call before execution.

    This prevents:
    SQL A -> empty result -> same SQL A again

    It still allows:
    SQL A -> SQL B
    SQL A -> API A
    API A -> API B
    """

    action_name = action.get("action")

    if action_name not in {"sql_query", "api_call"}:
        return None

    proposed_signature = build_tool_call_signature(action)

    if not proposed_signature:
        return None

    for previous_step in trace:
        previous_signature = build_executed_step_signature(previous_step)

        if proposed_signature == previous_signature:
            return {
                "ok": False,
                "target": "duplicate_tool_call",
                "errors": [
                    f"Duplicate {action_name} detected. This exact tool call was already executed."
                ],
                "repair_hint": (
                    "Do not repeat the same SQL/API call. "
                    "Choose a different SQL query, call a different valid API endpoint, "
                    "or return final_answer using the verified results already available."
                ),
            }

    return None


class TraceRouterAgent:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        executor: Optional[ToolExecutor] = None,
        config: Optional[AgentRunConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or AgentRunConfig()

        self.executor = executor or ToolExecutor(
            snapshot_dir=self.config.snapshot_dir,
            mock_api=self.config.mock_api,
        )

        self.api_index = load_api_index_if_available(self.config.api_index_path)

    def save_prompt_for_step(
        self,
        messages: List[Dict[str, str]],
        run_id: str,
        step: int,
    ) -> None:
        if not self.config.save_prompts:
            return

        out_dir = Path(self.config.prompt_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_json(
            messages,
            out_dir / f"{run_id}_step_{step:02d}_prompt.json",
        )
        
    def check_route_aware_policy(
        self,
        action: Dict[str, Any],
        metadata: Dict[str, Any],
        trace: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return route_policy_check(
            action=action,
            metadata=metadata,
            trace=trace,
            min_score=self.config.route_confidence_min_score,
            min_margin=self.config.route_confidence_min_margin,
            min_tools_before_final_when_uncertain=self.config.min_tools_before_final_when_uncertain,
        )

    def check_trajectory_budget(
        self,
        action: Dict[str, Any],
        metadata: Dict[str, Any],
        trace: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        budget = metadata.get("tool_budget", {}) or {}

        if action.get("action") == "sql_query":
            used_sql = count_trace_actions(trace, "sql_query")
            max_sql = budget.get("max_sql_calls", 0)

            if used_sql >= max_sql:
                return {
                    "ok": False,
                    "target": "sql",
                    "errors": [
                        f"SQL budget exceeded. Used {used_sql}, max allowed {max_sql}."
                    ],
                    "repair_hint": "Do not call SQL again. Use available results or produce final_answer.",
                }

        if action.get("action") == "api_call":
            used_api = count_trace_actions(trace, "api_call")
            max_api = budget.get("max_api_calls", 0)

            if used_api >= max_api:
                return {
                    "ok": False,
                    "target": "api",
                    "errors": [
                        f"API budget exceeded. Used {used_api}, max allowed {max_api}."
                    ],
                    "repair_hint": "Do not call API again. Use available results or produce final_answer.",
                }

        return None

    def execute_action(
        self,
        action: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        if action["action"] == "sql_query":
            sql = action.get("sql", "")
            execution = self.executor.execute_sql(sql)

            return {
                "step": step_number,
                "action": "sql_query",
                "sql": sql,
                "results": execution.get("rows", []),
                "status": execution.get("status"),
                "warnings": execution.get("warnings", []),
                "error": execution.get("error"),
            }

        if action["action"] == "api_call":
            method = action.get("method", "GET")
            url = action.get("url", "")
            params = action.get("params", {}) or {}
            headers = action.get("headers", {}) or {}
            body = action.get("body")

            execution = self.executor.call_api(
                method=method,
                url=url,
                params=params,
                headers=headers,
                body=body,
            )

            return {
                "step": step_number,
                "action": "api_call",
                "api_call": {
                    "method": method,
                    "url": url,
                    "params": params,
                    "status": execution.get("status"),
                    "status_code": execution.get("status_code"),
                    "result_preview": execution.get("result_preview", []),
                    "error": execution.get("error"),
                },
            }

        raise ValueError(f"Unsupported executable action: {action.get('action')}")

    def run(
        self,
        query: str,
        metadata: Dict[str, Any],
        run_id: str = "sample_run",
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        debug_events: List[Dict[str, Any]] = []

        verifier_feedback: Optional[Dict[str, Any]] = None
        repair_attempts = 0

        for step in range(1, self.config.max_steps + 1):
            messages = build_prompt_messages(
                query=query,
                metadata=metadata,
                tool_results=tool_results,
                verifier_feedback=verifier_feedback,
            )

            self.save_prompt_for_step(messages, run_id=run_id, step=step)

            raw_output = self.llm_client.generate(messages)

            debug_events.append(
                {
                    "step": step,
                    "event": "llm_output",
                    "raw_output": raw_output,
                }
            )

            try:
                action = parse_json_action(raw_output)
                action = normalize_api_action(action)
                
                route_policy_error = self.check_route_aware_policy(
                    action=action,
                    metadata=metadata,
                    trace=trace,
                )

                if route_policy_error:
                    verifier_feedback = route_policy_error

                    debug_events.append(
                        {
                            "step": step,
                            "event": "route_policy_rejected",
                            "proposed_action": action,
                            "feedback": verifier_feedback,
                        }
                    )

                    repair_attempts += 1

                    if repair_attempts > self.config.max_repair_attempts:
                        break

                    continue

            except Exception as e:
                verifier_feedback = {
                    "ok": False,
                    "target": "llm_output",
                    "errors": [str(e)],
                    "repair_hint": "Return exactly one valid JSON object with action sql_query, api_call, or final_answer.",
                }

                debug_events.append(
                    {
                        "step": step,
                        "event": "json_parse_failed",
                        "feedback": verifier_feedback,
                    }
                )

                repair_attempts += 1

                if repair_attempts > self.config.max_repair_attempts:
                    break

                continue

            if action.get("action") == "final_answer":
                llm_answer = action.get("answer", "")

                if self.config.use_answer_builder:
                    answer = build_final_answer(
                        query=query,
                        metadata=metadata,
                        trace=trace,
                        llm_answer=llm_answer,
                    )
                else:
                    answer = llm_answer

                return {
                    "query": query,
                    "trace": trace,
                    "answer": answer,
                    "status": "success",
                    "debug_events": debug_events,
                }

            budget_error = self.check_trajectory_budget(
                action=action,
                metadata=metadata,
                trace=trace,
            )

            if budget_error:
                verifier_feedback = budget_error
                debug_events.append(
                    {
                        "step": step,
                        "event": "budget_rejected",
                        "proposed_action": action,
                        "feedback": verifier_feedback,
                    }
                )

                repair_attempts += 1

                if repair_attempts > self.config.max_repair_attempts:
                    break

                continue


            duplicate_error = check_duplicate_tool_call(
                action=action,
                trace=trace,
            )

            if duplicate_error:
                verifier_feedback = duplicate_error
                debug_events.append(
                    {
                        "step": step,
                        "event": "duplicate_tool_call_rejected",
                        "proposed_action": action,
                        "feedback": verifier_feedback,
                    }
                )

                repair_attempts += 1

                if repair_attempts > self.config.max_repair_attempts:
                    break

                continue


            verification = validate_tool_call(
                tool_call=action,
                metadata=metadata,
                api_index=self.api_index,
            )

            if not verification.ok:
                verifier_feedback = verification.to_dict()

                debug_events.append(
                    {
                        "step": step,
                        "event": "verifier_rejected",
                        "proposed_action": action,
                        "feedback": verifier_feedback,
                    }
                )

                repair_attempts += 1

                if repair_attempts > self.config.max_repair_attempts:
                    break

                continue

            executed = self.execute_action(
                action=action,
                step_number=len(trace) + 1,
            )

            trace.append(executed)
            tool_results.append(executed)

            debug_events.append(
                {
                    "step": step,
                    "event": "tool_executed",
                    "action": action.get("action"),
                    "status": executed.get("status") or executed.get("api_call", {}).get("status"),
                }
            )

            verifier_feedback = None
            repair_attempts = 0

        return {
            "query": query,
            "trace": trace,
            "answer": "Unable to complete the query within the allowed agent steps.",
            "status": "failed",
            "debug_events": debug_events,
        }


def save_agent_run(result: Dict[str, Any], output_path: str | Path) -> None:
    save_json(result, output_path)


if __name__ == "__main__":
    from src.prompt_builder import load_json

    metadata = load_json("data/sample_metadata.json")
    query = metadata["query"]

    llm = StaticLLMClient(
        actions=[
            {
                "action": "sql_query",
                "sql": """
                SELECT CAMPAIGN.NAME AS CAMPAIGNNAME,
                       CAMPAIGN.CAMPAIGNID
                FROM DIM_CAMPAIGN AS CAMPAIGN
                LIMIT 10
                """,
            },
            {
                "action": "api_call",
                "method": "GET",
                "url": "/ajo/journey",
                "params": {"pageSize": "10"},
            },
            {
                "action": "final_answer",
                "answer": "The available journeys are listed in the SQL/API results.",
            },
        ]
    )

    config = AgentRunConfig(
        max_steps=5,
        mock_api=True,
    )

    agent = TraceRouterAgent(
        llm_client=llm,
        config=config,
    )

    result = agent.run(
        query=query,
        metadata=metadata,
        run_id="sample_journey",
    )

    save_agent_run(result, "data/sample_agent_run.json")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nSaved to: data/sample_agent_run.json")