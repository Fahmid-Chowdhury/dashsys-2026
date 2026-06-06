from src.metadata_generator import generate_metadata, print_metadata_summary, save_json

TEST_DIRECTORY = "data/test_metadata"

def main():
    test_queries = [
        "How many successful batches are there?",
        "List all journeys",
        "Show me all tags in this sandbox",
        "Which segment evaluation jobs are queued?",
        "Show datasets that use the same schema",
        "Give me recently modified destinations",
    ]

    for i, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 80)
        print(f"TEST {i}")
        print("=" * 80)

        metadata = generate_metadata(query)
        print_metadata_summary(metadata)

        output_path = f"{TEST_DIRECTORY}/test_metadata_{i}.json"
        save_json(metadata, output_path)
        print(f"\nSaved metadata to: {output_path}")


if __name__ == "__main__":
    main()