import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def get_selected_intent(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("intent", "unknown")).lower()


def get_selected_domain(metadata: Dict[str, Any]) -> str:
    domain = metadata.get("domain", {})

    if isinstance(domain, dict):
        return str(domain.get("selected", "unknown")).lower()

    return str(domain).lower()


def extract_sql_steps(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        step for step in trace
        if step.get("action") == "sql_query"
    ]


def extract_api_steps(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        step for step in trace
        if step.get("action") == "api_call"
    ]


def get_sql_rows(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for step in extract_sql_steps(trace):
        if step.get("status") == "success":
            rows.extend(step.get("results", []) or [])

    return rows


def get_api_call(step: Dict[str, Any]) -> Dict[str, Any]:
    return step.get("api_call", {}) or {}


def get_api_payload(step: Dict[str, Any]) -> Any:
    api_call = get_api_call(step)
    return api_call.get("result_preview")


def normalize_api_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Supports common Adobe API response shapes:

    1. AJO:
       {
         "results": [...],
         "pagination": {...}
       }

    2. Catalog:
       {
         "dataset_id_1": {...},
         "dataset_id_2": {...}
       }

    3. Generic list:
       [...]
    """

    if payload is None:
        return []

    if isinstance(payload, list):
        return [
            item if isinstance(item, dict) else {"value": item}
            for item in payload
        ]

    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [
                item if isinstance(item, dict) else {"value": item}
                for item in payload["results"]
            ]

        if isinstance(payload.get("items"), list):
            return [
                item if isinstance(item, dict) else {"value": item}
                for item in payload["items"]
            ]

        # Catalog APIs often return object keyed by id.
        catalog_like_items = []

        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("id", key)
                catalog_like_items.append(item)

        return catalog_like_items

    return []


def get_api_total_count(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    pagination = payload.get("pagination")

    if isinstance(pagination, dict):
        total = pagination.get("totalCount")
        if isinstance(total, int):
            return total

    if isinstance(payload.get("totalCount"), int):
        return payload["totalCount"]

    if isinstance(payload.get("count"), int):
        return payload["count"]

    return None


def summarize_identifier_row(row: Dict[str, Any]) -> str:
    """
    Creates a readable one-line summary from a SQL/API row.
    """

    name_keys = [
        "CAMPAIGNNAME",
        "campaign_name",
        "NAME",
        "name",
        "title",
        "displayName",
    ]

    id_keys = [
        "CAMPAIGNID",
        "campaign_id",
        "id",
        "_id",
        "datasetId",
        "dataSetId",
    ]

    status_keys = [
        "STATE",
        "state",
        "STATUS",
        "status",
    ]

    name = None
    identifier = None
    status = None

    for key in name_keys:
        if key in row and row[key] not in [None, ""]:
            name = row[key]
            break

    for key in id_keys:
        if key in row and row[key] not in [None, ""]:
            identifier = row[key]
            break

    for key in status_keys:
        if key in row and row[key] not in [None, ""]:
            status = row[key]
            break

    parts = []

    if name:
        parts.append(str(name))

    if identifier:
        parts.append(f"ID: {identifier}")

    if status:
        parts.append(f"status/state: {status}")

    if parts:
        return " — ".join(parts)

    # Fallback: show compact key-value preview.
    preview_items = list(row.items())[:4]
    return ", ".join(f"{k}: {v}" for k, v in preview_items)


def format_rows_as_bullets(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    if not rows:
        return "None."

    lines = []

    for row in rows[:max_rows]:
        lines.append(f"- {summarize_identifier_row(row)}")

    if len(rows) > max_rows:
        lines.append(f"- ... and {len(rows) - max_rows} more rows.")

    return "\n".join(lines)


def summarize_api_step(step: Dict[str, Any]) -> Dict[str, Any]:
    api_call = get_api_call(step)
    payload = get_api_payload(step)

    items = normalize_api_items(payload)
    total_count = get_api_total_count(payload)

    return {
        "method": api_call.get("method"),
        "url": api_call.get("url"),
        "params": api_call.get("params", {}),
        "status": api_call.get("status"),
        "status_code": api_call.get("status_code"),
        "items": items,
        "item_count": len(items),
        "total_count": total_count,
        "error": api_call.get("error"),
    }


def build_list_answer(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> str:
    domain = get_selected_domain(metadata)

    sql_rows = get_sql_rows(trace)
    api_steps = extract_api_steps(trace)
    api_summaries = [summarize_api_step(step) for step in api_steps]

    lines = []

    if sql_rows:
        lines.append(
            f"From the local SQL snapshot, I found {len(sql_rows)} {domain} record(s):"
        )
        lines.append(format_rows_as_bullets(sql_rows))
    else:
        lines.append(f"The local SQL snapshot returned no {domain} records.")

    if api_summaries:
        for api in api_summaries:
            endpoint = f"{api.get('method')} {api.get('url')}"
            status_code = api.get("status_code")
            total_count = api.get("total_count")
            item_count = api.get("item_count")

            if api.get("status") != "success":
                lines.append(
                    f"The live API call to {endpoint} failed"
                    f"{f' with HTTP {status_code}' if status_code else ''}."
                )

                if api.get("error"):
                    lines.append(f"API error: {api['error']}")

                continue

            if total_count is not None:
                lines.append(
                    f"The live API call to {endpoint} succeeded with HTTP {status_code} "
                    f"and reported {total_count} matching record(s)."
                )
            else:
                lines.append(
                    f"The live API call to {endpoint} succeeded with HTTP {status_code} "
                    f"and returned {item_count} preview item(s)."
                )

            if api.get("items"):
                lines.append("API preview items:")
                lines.append(format_rows_as_bullets(api["items"]))

        # Basic discrepancy detection.
        first_api = api_summaries[0]
        api_count = (
            first_api["total_count"]
            if first_api.get("total_count") is not None
            else first_api.get("item_count", 0)
        )

        if len(sql_rows) > 0 and api_count == 0:
            lines.append(
                "Discrepancy: the local SQL snapshot contains records, "
                "but the live API returned zero records for the current sandbox/query."
            )

        elif len(sql_rows) == 0 and api_count > 0:
            lines.append(
                "Discrepancy: the live API returned records, "
                "but the local SQL snapshot returned none."
            )

    return "\n\n".join(lines)


def build_count_answer(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> str:
    sql_rows = get_sql_rows(trace)
    api_steps = extract_api_steps(trace)
    api_summaries = [summarize_api_step(step) for step in api_steps]

    lines = []

    if sql_rows:
        first_row = sql_rows[0]

        count_value = None
        for key, value in first_row.items():
            if isinstance(value, int):
                count_value = value
                break

        if count_value is not None:
            lines.append(f"The SQL snapshot count is {count_value}.")
        else:
            lines.append(f"The SQL snapshot returned {len(sql_rows)} row(s).")
    else:
        lines.append("The SQL snapshot returned no rows.")

    for api in api_summaries:
        endpoint = f"{api.get('method')} {api.get('url')}"

        if api.get("status") != "success":
            lines.append(f"The API call to {endpoint} failed.")
            continue

        if api.get("total_count") is not None:
            lines.append(
                f"The API call to {endpoint} reported {api['total_count']} record(s)."
            )
        else:
            lines.append(
                f"The API call to {endpoint} returned {api['item_count']} preview item(s)."
            )

    return "\n\n".join(lines)


def build_generic_answer(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
    llm_answer: Optional[str] = None,
) -> str:
    sql_rows = get_sql_rows(trace)
    api_steps = extract_api_steps(trace)
    api_summaries = [summarize_api_step(step) for step in api_steps]

    lines = []

    if llm_answer:
        lines.append(llm_answer.strip())

    if sql_rows:
        lines.append(f"SQL returned {len(sql_rows)} row(s):")
        lines.append(format_rows_as_bullets(sql_rows))

    for api in api_summaries:
        endpoint = f"{api.get('method')} {api.get('url')}"

        if api.get("status") == "success":
            if api.get("total_count") is not None:
                lines.append(
                    f"API {endpoint} returned HTTP {api.get('status_code')} "
                    f"with totalCount={api.get('total_count')}."
                )
            else:
                lines.append(
                    f"API {endpoint} returned HTTP {api.get('status_code')} "
                    f"with {api.get('item_count')} preview item(s)."
                )
        else:
            lines.append(f"API {endpoint} failed.")

    if not lines:
        return "No verified SQL/API results were available to answer the query."

    return "\n\n".join(lines)


def build_final_answer(
    query: str,
    metadata: Dict[str, Any],
    trace: List[Dict[str, Any]],
    llm_answer: Optional[str] = None,
) -> str:
    intent = get_selected_intent(metadata)

    if intent in {"list", "search"}:
        return build_list_answer(
            query=query,
            metadata=metadata,
            trace=trace,
        )

    if intent in {"count", "count_lookup"}:
        return build_count_answer(
            query=query,
            metadata=metadata,
            trace=trace,
        )

    return build_generic_answer(
        query=query,
        metadata=metadata,
        trace=trace,
        llm_answer=llm_answer,
    )


if __name__ == "__main__":
    run = load_json("data/test_agent_run.json")
    metadata = load_json("data/sample_metadata.json")

    answer = build_final_answer(
        query=run["query"],
        metadata=metadata,
        trace=run["trace"],
        llm_answer=run.get("answer"),
    )

    print(answer)

    output = dict(run)
    output["answer"] = answer

    save_json(output, "data/test_agent_run_with_built_answer.json")
    print("\nSaved to: data/test_agent_run_with_built_answer.json")