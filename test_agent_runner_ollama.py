import json
import os

from src.agent_runner import (
    AgentRunConfig,
    OllamaLLMClient,
    TraceRouterAgent,
    save_agent_run,
)
from src.prompt_builder import load_json


def main():
    metadata = load_json("data/sample_metadata.json")
    query = metadata["query"]

    model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    llm = OllamaLLMClient(
        model=model,
        temperature=0.0,
        num_ctx=8192,
        use_json_schema=True,
    )

    config = AgentRunConfig(
        max_steps=5,
        max_repair_attempts=2,
        snapshot_dir="DBSnapshot",
        api_index_path="data/api_index_enriched.json",
        mock_api=False,
        save_prompts=True,
        use_answer_builder=True,
    )

    agent = TraceRouterAgent(
        llm_client=llm,
        config=config,
    )

    result = agent.run(
        query=query,
        metadata=metadata,
        run_id="ollama_journey_test",
    )

    save_agent_run(result, "data/test_ollama_agent_run.json")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nSaved to: data/test_ollama_agent_run.json")

    if result["status"] != "success":
        print("\nOllama run did not complete successfully.")
        print("Check debug_events and saved prompts in data/agent_prompts/.")


if __name__ == "__main__":
    main()