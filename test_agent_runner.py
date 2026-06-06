import json

from src.agent_runner import (
    AgentRunConfig,
    StaticLLMClient,
    TraceRouterAgent,
    save_agent_run,
)
from src.prompt_builder import load_json


def main():
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
                "answer": "The available journeys are listed in the verified SQL and API results.",
            },
        ]
    )

    config = AgentRunConfig(
        max_steps=5,
        snapshot_dir="DBSnapshot",
        api_index_path="data/api_index_enriched.json",
        mock_api=False,
        save_prompts=True,
    )

    agent = TraceRouterAgent(
        llm_client=llm,
        config=config,
    )

    result = agent.run(
        query=query,
        metadata=metadata,
        run_id="test_agent_runner",
    )

    assert result["status"] == "success"
    assert result["query"] == query
    assert len(result["trace"]) == 2
    assert result["trace"][0]["action"] == "sql_query"
    assert result["trace"][1]["action"] == "api_call"
    assert result["answer"]

    save_agent_run(result, "data/test_agent_run.json")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nAgent runner test passed.")
    print("Saved to: data/test_agent_run.json")


if __name__ == "__main__":
    main()