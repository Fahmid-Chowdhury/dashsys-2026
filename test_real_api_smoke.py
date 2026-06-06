import json

from src.tool_executor import ToolExecutor


def compact_print(name, result):
    print("\n" + name)
    print("=" * len(name))
    print("status:", result.get("status"))
    print("status_code:", result.get("status_code"))
    print("url:", result.get("full_url"))

    preview = result.get("result_preview")
    print("preview:")
    print(json.dumps(preview, indent=2, ensure_ascii=False)[:1500])


def main():
    executor = ToolExecutor(mock_api=False)

    tests = [
        (
            "Catalog datasets",
            "GET",
            "/data/foundation/catalog/dataSets",
            {"limit": "2"},
        ),
        (
            "Catalog batches",
            "GET",
            "/data/foundation/catalog/batches",
            {"limit": "2"},
        ),
        (
            "AJO journeys",
            "GET",
            "/ajo/journey",
            {"pageSize": "5", "page": "0"},
        ),
    ]

    for name, method, url, params in tests:
        result = executor.call_api(
            method=method,
            url=url,
            params=params,
        )
        compact_print(name, result)


if __name__ == "__main__":
    main()