import json

from src.prompt_builder import (
    build_prompt_messages,
    load_json,
    print_prompt_preview,
    save_prompt_messages,
)


def main():
    metadata = load_json("data/sample_metadata.json")
    query = metadata["query"]

    messages = build_prompt_messages(
        query=query,
        metadata=metadata,
        tool_results=None,
        verifier_feedback=None,
    )

    assert isinstance(messages, list)
    assert len(messages) == 2

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "metadata-grounded SQL/API planning agent" in system_prompt
    assert "USER QUERY" in user_prompt
    assert "ALLOWED SQL TABLES" in user_prompt
    assert "ALLOWED API ENDPOINTS" in user_prompt
    assert "dim_campaign" in user_prompt
    assert "GET /ajo/journey" in user_prompt
    assert "Return only JSON" in user_prompt or "Return only one valid JSON object" in user_prompt

    save_prompt_messages(messages, "data/sample_prompt_messages_test.json")

    print_prompt_preview(messages, max_chars=2500)
    print("\nPrompt builder test passed.")
    print("Saved to: data/sample_prompt_messages_test.json")


if __name__ == "__main__":
    main()