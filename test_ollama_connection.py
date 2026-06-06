import json
import os

from src.agent_runner import OllamaLLMClient


def main():
    model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    llm = OllamaLLMClient(
        model=model,
        temperature=0.0,
        use_json_schema=True,
    )

    messages = [
        {
            "role": "system",
            "content": "Return only one JSON object.",
        },
        {
            "role": "user",
            "content": """
            Return a final answer JSON object.

            The JSON must be:
            {
            "action": "final_answer",
            "answer": "Ollama connection works."
            }
            """,
        },
    ]

    output = llm.generate(messages)

    print("Raw Ollama output")
    print("=================")
    print(output)

    parsed = json.loads(output)

    assert parsed["action"] == "final_answer"
    assert "answer" in parsed

    print("\nOllama connection test passed.")


if __name__ == "__main__":
    main()