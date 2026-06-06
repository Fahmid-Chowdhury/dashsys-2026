from src.api_index import (
    build_api_index_from_example_index,
    parse_gold_api_call,
    save_json,
)


def main():
    print("Parser test")
    print("===========")

    api = "GET /data/foundation/catalog/batches?limit=10&status=success"
    parsed = parse_gold_api_call(api)

    print(parsed)

    print("\nBuilding API index")
    print("==================")

    api_index = build_api_index_from_example_index("data/example_index.json")

    print("Endpoint count:", api_index["endpoint_count"])

    for endpoint, record in list(api_index["endpoints"].items())[:10]:
        print()
        print("Endpoint:", endpoint)
        print("Domain:", record["domain"])
        print("Params:", record["param_names"])
        print("Routes seen:", record["routes_seen"])

    save_json(api_index, "data/api_index_test.json")
    print("\nSaved test API index to: data/api_index_test.json")


if __name__ == "__main__":
    main()