from src.answer_builder import (
    build_final_answer,
    load_json,
    save_json,
)


def main():
    run = load_json("data/test_agent_run.json")
    metadata = load_json("data/sample_metadata.json")

    answer = build_final_answer(
        query=run["query"],
        metadata=metadata,
        trace=run["trace"],
        llm_answer=run.get("answer"),
    )

    print("Generated answer")
    print("================")
    print(answer)

    assert "Birthday Message" in answer
    assert "Gold Tier Welcome Email" in answer
    assert "Discrepancy" in answer or "zero records" in answer

    updated = dict(run)
    updated["answer"] = answer

    save_json(updated, "data/test_agent_run_with_built_answer.json")

    print("\nAnswer builder test passed.")
    print("Saved to: data/test_agent_run_with_built_answer.json")


if __name__ == "__main__":
    main()