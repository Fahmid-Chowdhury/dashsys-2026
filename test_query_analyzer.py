import json
from src.query_analyzer import analyze_query


def main():
    queries = [
        "List all journeys",
        "List all segment audiences connected to the destination named 'Lives on East Coast', showing audienceId, name, totalProfiles, createdTime, updatedTime, and used in other audience count for each audience.",
        "Show ingestion record counts and batch success counts for the last 60 days.",
    ]

    for query in queries:
        print("=" * 80)
        print(query)
        print("=" * 80)

        analysis = analyze_query(query)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()