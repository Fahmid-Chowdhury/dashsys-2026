import json

from src.metadata_generator import generate_metadata


def main():
    queries = [
        "List all journeys",
        "List all segment audiences connected to the destination named 'Lives on East Coast', showing audienceId, name, totalProfiles, createdTime, updatedTime, and used in other audience count for each audience. Remove any row limit from the results.",
        "Show ingestion record counts and batch success counts for the last 60 days.",
    ]

    for query in queries:
        print("=" * 100)
        print(query)
        print("=" * 100)

        metadata = generate_metadata(query)

        summary = {
            "query": metadata["query"],
            "query_analysis": metadata.get("query_analysis"),
            "route": metadata.get("route"),
            "domain": metadata.get("domain"),
            "tool_budget": metadata.get("tool_budget"),
            "similar_example_count": len(metadata.get("similar_examples", [])),
            "similar_examples": [
                {
                    "query": ex.get("query"),
                    "score": ex.get("score"),
                    "sources": ex.get("retrieval_sources"),
                }
                for ex in metadata.get("similar_examples", [])
            ],
        }

        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()