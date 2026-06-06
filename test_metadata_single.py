import json
from src.metadata_generator import generate_metadata


def main():
    queries = [
        "When was the journey 'Gold Tier Welcome Email' published?",
        "List all datasets that use the schema 'weRetail: CRM'.",
        "Show me the details of the tag named 'AI-Generated'.",
    ]

    for query in queries:
        print("=" * 100)
        print(query)
        print("=" * 100)

        metadata = generate_metadata(query)

        summary = {
            "query": metadata.get("query"),
            "query_analysis": metadata.get("query_analysis"),
            "route": metadata.get("route"),
            "domain": metadata.get("domain"),
            "intent": metadata.get("intent"),
            "tool_budget": metadata.get("tool_budget"),
            "allowed_tables": list((metadata.get("allowed_tables") or {}).keys()),
            "allowed_api_endpoints": [
                e.get("endpoint")
                for e in metadata.get("allowed_api_endpoints", [])
            ],
            "similar_examples": [
                {
                    "query": ex.get("query"),
                    "score": ex.get("score"),
                    "route": ex.get("route"),
                    "domain": ex.get("domain"),
                    "retrieval_sources": ex.get("retrieval_sources"),
                }
                for ex in metadata.get("similar_examples", [])
            ],
        }

        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()