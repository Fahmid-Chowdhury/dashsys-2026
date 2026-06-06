from src.hybrid_retriever import HybridExampleRetriever


def main():
    retriever = HybridExampleRetriever()

    test_queries = [
        "How many successful batches are there?",
        "List all journeys",
        "Show me all tags in this sandbox",
        "Which segment evaluation jobs are queued?",
        "Show datasets that use the same schema",
        "Give me recently modified destinations",
    ]

    for query in test_queries:
        print("\n" + "=" * 90)
        print("USER QUERY:", query)

        results = retriever.search(query, top_k=5)

        for result in results:
            print(
                f"\nRank {result['rank']}"
                f"\nCombined score: {result['combined_score']:.4f}"
                f"\nTF-IDF score: {result['tfidf_score']:.4f}"
                f"\nDense score: {result['dense_score']:.4f}"
                f"\nStructured score: {result['structured_score']:.4f}"
                f"\nRoute: {result['route']}"
                f"\nDomain: {result['domain']}"
                f"\nIntent: {result['intent']}"
                f"\nSimilar query: {result['query']}"
                f"\nSQL tables: {result['sql_tables']}"
                f"\nAPI endpoints: {result['api_endpoints']}"
                f"\nAPI params: {result['api_params']}"
            )

        print("\nRoute votes:")
        print(retriever.route_votes(results))

        print("\nDomain votes:")
        print(retriever.domain_votes(results))


if __name__ == "__main__":
    main()