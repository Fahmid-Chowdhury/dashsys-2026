import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_gold_api_call(api_call: str) -> Dict[str, Any]:
    """
    Parses:
    GET /data/foundation/catalog/batches?limit=10&status=success

    Into:
    {
      "method": "GET",
      "path": "/data/foundation/catalog/batches",
      "endpoint": "GET /data/foundation/catalog/batches",
      "params": {"limit": "10", "status": "success"}
    }
    """

    api_call = api_call.strip()

    if not api_call:
        return {}

    parts = api_call.split(maxsplit=1)

    if len(parts) != 2:
        return {
            "method": "UNKNOWN",
            "path": api_call,
            "endpoint": api_call,
            "params": {}
        }

    method, rest = parts
    method = method.upper()

    if "?" in rest:
        path, query_string = rest.split("?", 1)
        raw_params = parse_qs(query_string, keep_blank_values=True)

        params = {
            key: values[0] if values else ""
            for key, values in raw_params.items()
        }
    else:
        path = rest
        params = {}

    endpoint = f"{method} {path}"

    return {
        "method": method,
        "path": path,
        "endpoint": endpoint,
        "params": params,
    }


def infer_api_domain(path: str) -> str:
    p = path.lower()

    rules = [
        ("journey", ["/ajo/journey"]),
        ("batch", ["/catalog/batches", "/catalog/batch"]),
        ("dataset", ["/catalog/datasets", "/catalog/datasets", "/catalog/dataSets".lower()]),
        ("schema", ["/schemaregistry", "/schemas"]),
        ("destination", ["/flowservice/flows"]),
        ("dataflow", ["/flowservice/flows", "/runs"]),
        ("segment", ["/ups/audiences"]),
        ("segment_job", ["/segment/jobs", "/segmentjobs"]),
        ("segment_definition", ["/segment/definitions", "/segment/definitions"]),
        ("tag", ["/tags"]),
        ("merge_policy", ["/config/mergepolicies", "/mergepolicies"]),
        ("metric", ["/observability", "/metrics"]),
    ]

    for domain, keywords in rules:
        if any(k.lower() in p for k in keywords):
            return domain

    return "general"


def merge_endpoint_record(
    api_index: Dict[str, Any],
    parsed: Dict[str, Any],
    source_query: str,
    route: str | None = None,
    example_domain: str | None = None,
) -> None:
    if not parsed:
        return

    endpoint = parsed["endpoint"]
    method = parsed["method"]
    path = parsed["path"]
    params = parsed.get("params", {})

    if endpoint not in api_index:
        api_index[endpoint] = {
            "endpoint": endpoint,
            "method": method,
            "path": path,
            "domain": example_domain or infer_api_domain(path),
            "observed_params": {},
            "param_names": [],
            "routes_seen": [],
            "source_queries": [],
            "source": "gold_examples",
        }

    record = api_index[endpoint]

    for key, value in params.items():
        if key not in record["observed_params"]:
            record["observed_params"][key] = []

        if value not in record["observed_params"][key]:
            record["observed_params"][key].append(value)

    record["param_names"] = sorted(record["observed_params"].keys())

    if route and route not in record["routes_seen"]:
        record["routes_seen"].append(route)

    if source_query and source_query not in record["source_queries"]:
        record["source_queries"].append(source_query)


def build_api_index_from_example_index(example_index_path: str | Path) -> Dict[str, Any]:
    examples = load_json(example_index_path)

    api_index: Dict[str, Any] = {}

    for example in examples:
        query = example.get("query", "")
        route = example.get("route")
        domain = example.get("domain")

        gold_api = example.get("gold_api", []) or []

        for api_call in gold_api:
            parsed = parse_gold_api_call(api_call)
            merge_endpoint_record(
                api_index=api_index,
                parsed=parsed,
                source_query=query,
                route=route,
                example_domain=domain,
            )

    return {
        "endpoint_count": len(api_index),
        "endpoints": dict(sorted(api_index.items())),
    }


def print_api_index_summary(api_index: Dict[str, Any]) -> None:
    print("API index summary")
    print("=================")
    print(f"Endpoint count: {api_index['endpoint_count']}")

    print("\nEndpoints:")
    for endpoint, record in api_index["endpoints"].items():
        params = ", ".join(record["param_names"]) if record["param_names"] else "none"
        routes = ", ".join(record["routes_seen"]) if record["routes_seen"] else "none"

        print(f"  - {endpoint}")
        print(f"    domain: {record['domain']}")
        print(f"    params: {params}")
        print(f"    routes: {routes}")


if __name__ == "__main__":
    api_index = build_api_index_from_example_index("data/example_index.json")
    save_json(api_index, "data/api_index.json")

    print_api_index_summary(api_index)
    print("\nSaved to: data/api_index.json")