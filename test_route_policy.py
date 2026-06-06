from src.prompt_builder import load_json
from src.route_policy import (
    check_route_aware_policy,
    get_prompt_allowed_actions,
    is_route_confident,
)


def main():
    metadata = load_json("data/sample_metadata.json")

    assert is_route_confident(metadata) is True

    empty_trace = []

    # For confident SQL_PLUS_API, final answer should not be allowed before tools.
    result = check_route_aware_policy(
        action={"action": "final_answer", "answer": "Done."},
        metadata=metadata,
        trace=empty_trace,
    )

    assert result is not None
    assert result["ok"] is False
    print("Premature final answer rejected.")

    # API first is allowed. We are not forcing exact order.
    result = check_route_aware_policy(
        action={
            "action": "api_call",
            "method": "GET",
            "url": "/ajo/journey",
            "params": {"pageSize": "10"},
        },
        metadata=metadata,
        trace=empty_trace,
    )

    assert result is None
    print("API first allowed.")

    # After API succeeds, final answer is still not allowed because SQL is missing.
    api_trace = [
        {
            "step": 1,
            "action": "api_call",
            "api_call": {
                "method": "GET",
                "url": "/ajo/journey",
                "params": {"pageSize": "10"},
                "status": "success",
                "status_code": 200,
                "result_preview": {"results": []},
            },
        }
    ]

    result = check_route_aware_policy(
        action={"action": "final_answer", "answer": "Done."},
        metadata=metadata,
        trace=api_trace,
    )

    assert result is not None
    assert "sql_query" in str(result["errors"])
    print("Final answer after only API rejected.")

    allowed = get_prompt_allowed_actions(
        metadata=metadata,
        trace=api_trace,
    )

    assert allowed == ["sql_query"]
    print("After API, prompt correctly allows only SQL.")

    full_trace = api_trace + [
        {
            "step": 2,
            "action": "sql_query",
            "sql": "SELECT NAME FROM DIM_CAMPAIGN LIMIT 10",
            "results": [{"NAME": "Birthday Message"}],
            "status": "success",
        }
    ]

    result = check_route_aware_policy(
        action={"action": "final_answer", "answer": "Done."},
        metadata=metadata,
        trace=full_trace,
    )

    assert result is None
    print("Final answer after SQL + API allowed.")

    allowed = get_prompt_allowed_actions(
        metadata=metadata,
        trace=full_trace,
    )

    assert allowed == ["final_answer"]
    print("After SQL + API, prompt correctly allows final_answer.")

    print("\nRoute policy test passed.")


if __name__ == "__main__":
    main()