import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head"}


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


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def resolve_ref(openapi_doc: Dict[str, Any], ref: str) -> Dict[str, Any]:
    """
    Resolves local references like:
    #/parameters/x-api-key
    #/components/parameters/foo
    """
    if not ref.startswith("#/"):
        return {}

    parts = ref[2:].split("/")
    current: Any = openapi_doc

    for part in parts:
        if not isinstance(current, dict):
            return {}
        current = current.get(part)

    return current if isinstance(current, dict) else {}


def extract_base_path_from_servers(openapi_doc: Dict[str, Any]) -> str:
    """
    Supports both:
    OpenAPI 3.x: servers[0].url
    Swagger 2.0: basePath
    """

    # Swagger/OpenAPI 2.0
    if openapi_doc.get("basePath"):
        base_path = openapi_doc["basePath"]
        if not base_path.startswith("/"):
            base_path = "/" + base_path
        return base_path.rstrip("/")

    # OpenAPI 3.x
    servers = openapi_doc.get("servers", []) or []

    if not servers:
        return ""

    url = servers[0].get("url", "")

    marker = ".adobe.io"

    if marker in url:
        base_path = url.split(marker, 1)[1]
    else:
        base_path = url

    if "}" in base_path:
        base_path = base_path.split("}", 1)[-1]

    if not base_path:
        return ""

    if not base_path.startswith("/"):
        base_path = "/" + base_path

    return base_path.rstrip("/")


def normalize_openapi_path(path: str, base_path: str = "") -> str:
    if not path.startswith("/"):
        path = "/" + path

    path = path.rstrip("/") if path != "/" else path

    if base_path:
        base_path = base_path.rstrip("/")

        if not path.startswith(base_path):
            path = base_path + path

    return path


def infer_domain_from_path(path: str) -> str:
    p = path.lower()

    rules = [
        ("journey", ["/ajo/journey"]),
        ("batch", ["/catalog/batches", "/catalog/batch"]),
        ("dataset", ["/catalog/datasets", "/catalog/datasets"]),
        ("schema", ["/schemaregistry", "/schemas"]),
        ("destination", ["/flowservice/flows"]),
        ("dataflow", ["/flowservice/flows", "/runs"]),
        ("segment", ["/ups/audiences"]),
        ("segment_job", ["/segment/jobs", "/segmentjobs", "/jobs"]),
        ("segment_definition", ["/segment/definitions"]),
        ("tag", ["/unifiedtags", "/tags"]),
        ("merge_policy", ["/mergepolicies", "/merge-policies"]),
        ("metric", ["/observability", "/metrics"]),
        ("data_access", ["/data/foundation/export"]),
    ]

    for domain, keywords in rules:
        if any(keyword in p for keyword in keywords):
            return domain

    return "general"


def extract_parameters(
    openapi_doc: Dict[str, Any],
    path_level_params: List[Dict[str, Any]],
    operation_params: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_params = []

    for param in path_level_params + operation_params:
        if "$ref" in param:
            resolved = resolve_ref(openapi_doc, param["$ref"])
            if resolved:
                all_params.append(resolved)
        else:
            all_params.append(param)

    query_params = []
    path_params = []
    header_params = []
    body_params = []
    required_params = []

    for param in all_params:
        name = param.get("name")
        location = param.get("in")
        required = bool(param.get("required", False))

        if not name or not location:
            continue

        record = {
            "name": name,
            "required": required,
            "schema": param.get("schema", {}),
            "type": param.get("type"),
            "description": param.get("description", ""),
        }

        if location == "query":
            query_params.append(record)
        elif location == "path":
            path_params.append(record)
        elif location == "header":
            header_params.append(record)
        elif location == "body":
            body_params.append(record)

        if required:
            required_params.append(name)

    return {
        "query_params": query_params,
        "path_params": path_params,
        "header_params": header_params,
        "body_params": body_params,
        "required_params": sorted(set(required_params)),
        "query_param_names": sorted({p["name"] for p in query_params}),
        "path_param_names": sorted({p["name"] for p in path_params}),
        "header_param_names": sorted({p["name"] for p in header_params}),
        "body_param_names": sorted({p["name"] for p in body_params}),
    }


def extract_request_body(
    operation: Dict[str, Any],
    body_params: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # OpenAPI 3.x
    request_body = operation.get("requestBody")

    if request_body:
        content = request_body.get("content", {})
        content_types = list(content.keys())

        schema = {}

        if content_types:
            first_content = content.get(content_types[0], {})
            schema = first_content.get("schema", {}) or {}

        return {
            "required": bool(request_body.get("required", False)),
            "content_types": content_types,
            "schema": schema,
        }

    # Swagger/OpenAPI 2.0 body parameter
    body_params = body_params or []

    if body_params:
        first_body = body_params[0]

        return {
            "required": bool(first_body.get("required", False)),
            "content_types": ["application/json"],
            "schema": first_body.get("schema", {}),
        }

    return {
        "required": False,
        "content_types": [],
        "schema": {},
    }


def extract_response_codes(operation: Dict[str, Any]) -> List[str]:
    responses = operation.get("responses", {}) or {}
    return sorted(str(code) for code in responses.keys())


def extract_openapi_endpoints(openapi_doc: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    paths = openapi_doc.get("paths", {}) or {}
    base_path = extract_base_path_from_servers(openapi_doc)
    endpoints: Dict[str, Any] = {}

    for raw_path, path_item in paths.items():
        path = normalize_openapi_path(raw_path, base_path=base_path)
        if not isinstance(path_item, dict):
            continue

        path = normalize_openapi_path(raw_path, base_path=base_path)
        path_level_params = path_item.get("parameters", []) or []

        for method, operation in path_item.items():
            method_lower = method.lower()

            if method_lower not in HTTP_METHODS:
                continue

            if not isinstance(operation, dict):
                continue

            method_upper = method_lower.upper()
            endpoint = f"{method_upper} {path}"

            param_info = extract_parameters(
                openapi_doc=openapi_doc,
                path_level_params=path_level_params,
                operation_params=operation.get("parameters", []) or [],
            )

            request_body = extract_request_body(
                operation=operation,
                body_params=param_info.get("body_params", []),
            )

            endpoints[endpoint] = {
                "endpoint": endpoint,
                "method": method_upper,
                "path": path,
                "domain": infer_domain_from_path(path),
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
                "operation_id": operation.get("operationId", ""),
                "query_params": param_info["query_params"],
                "path_params": param_info["path_params"],
                "header_params": param_info["header_params"],
                "query_param_names": param_info["query_param_names"],
                "path_param_names": param_info["path_param_names"],
                "header_param_names": param_info["header_param_names"],
                "required_params": param_info["required_params"],
                "request_body": request_body,
                "response_codes": extract_response_codes(operation),
                "source": "openapi",
                "source_file": source_file,
            }

    return endpoints


def load_openapi_specs(spec_dir: str | Path) -> Dict[str, Any]:
    spec_dir = Path(spec_dir)

    if not spec_dir.exists():
        return {
            "spec_count": 0,
            "endpoints": {},
            "warnings": [f"OpenAPI directory not found: {spec_dir}"],
        }

    endpoints: Dict[str, Any] = {}
    warnings: List[str] = []

    yaml_files = list(spec_dir.glob("*.yaml")) + list(spec_dir.glob("*.yml"))

    for yaml_path in yaml_files:
        try:
            doc = load_yaml(yaml_path)
            extracted = extract_openapi_endpoints(doc, source_file=yaml_path.name)
            endpoints.update(extracted)

        except Exception as e:
            warnings.append(f"Failed to parse {yaml_path.name}: {str(e)}")

    return {
        "spec_count": len(yaml_files),
        "endpoints": endpoints,
        "warnings": warnings,
    }


def merge_example_and_openapi_index(
    example_api_index: Dict[str, Any],
    openapi_result: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    example_endpoints = example_api_index.get("endpoints", {}) or {}
    openapi_endpoints = openapi_result.get("endpoints", {}) or {}

    all_keys = sorted(set(example_endpoints.keys()) | set(openapi_endpoints.keys()))

    for endpoint in all_keys:
        example_record = example_endpoints.get(endpoint)
        openapi_record = openapi_endpoints.get(endpoint)

        if example_record and openapi_record:
            merged[endpoint] = {
                **example_record,
                "source": "gold_examples+openapi",
                "openapi": {
                    "summary": openapi_record.get("summary", ""),
                    "description": openapi_record.get("description", ""),
                    "operation_id": openapi_record.get("operation_id", ""),
                    "query_params": openapi_record.get("query_params", []),
                    "path_params": openapi_record.get("path_params", []),
                    "header_params": openapi_record.get("header_params", []),
                    "query_param_names": openapi_record.get("query_param_names", []),
                    "path_param_names": openapi_record.get("path_param_names", []),
                    "header_param_names": openapi_record.get("header_param_names", []),
                    "required_params": openapi_record.get("required_params", []),
                    "request_body": openapi_record.get("request_body", {}),
                    "response_codes": openapi_record.get("response_codes", []),
                    "source_file": openapi_record.get("source_file", ""),
                },
            }

        elif example_record:
            merged[endpoint] = {
                **example_record,
                "source": "gold_examples",
                "openapi": None,
            }

        else:
            if openapi_record is None:
                continue
            merged[endpoint] = {
                "endpoint": openapi_record["endpoint"],
                "method": openapi_record["method"],
                "path": openapi_record["path"],
                "domain": openapi_record["domain"],
                "observed_params": {},
                "param_names": openapi_record.get("query_param_names", []),
                "routes_seen": [],
                "source_queries": [],
                "source": "openapi",
                "openapi": {
                    "summary": openapi_record.get("summary", ""),
                    "description": openapi_record.get("description", ""),
                    "operation_id": openapi_record.get("operation_id", ""),
                    "query_params": openapi_record.get("query_params", []),
                    "path_params": openapi_record.get("path_params", []),
                    "header_params": openapi_record.get("header_params", []),
                    "query_param_names": openapi_record.get("query_param_names", []),
                    "path_param_names": openapi_record.get("path_param_names", []),
                    "header_param_names": openapi_record.get("header_param_names", []),
                    "required_params": openapi_record.get("required_params", []),
                    "request_body": openapi_record.get("request_body", {}),
                    "response_codes": openapi_record.get("response_codes", []),
                    "source_file": openapi_record.get("source_file", ""),
                },
            }

    return {
        "endpoint_count": len(merged),
        "gold_example_endpoint_count": len(example_endpoints),
        "openapi_endpoint_count": len(openapi_endpoints),
        "spec_count": openapi_result.get("spec_count", 0),
        "warnings": openapi_result.get("warnings", []),
        "endpoints": merged,
    }


def build_enriched_api_index(
    example_api_index_path: str | Path = "data/api_index.json",
    openapi_spec_dir: str | Path = "openapi_specs",
    output_path: str | Path = "data/api_index_enriched.json",
) -> Dict[str, Any]:
    example_api_index = load_json(example_api_index_path)
    openapi_result = load_openapi_specs(openapi_spec_dir)

    enriched = merge_example_and_openapi_index(
        example_api_index=example_api_index,
        openapi_result=openapi_result,
    )

    save_json(enriched, output_path)

    return enriched


def print_enriched_api_summary(api_index: Dict[str, Any]) -> None:
    print("Enriched API index summary")
    print("==========================")
    print(f"Spec files parsed: {api_index.get('spec_count', 0)}")
    print(f"Gold-example endpoints: {api_index.get('gold_example_endpoint_count', 0)}")
    print(f"OpenAPI endpoints: {api_index.get('openapi_endpoint_count', 0)}")
    print(f"Total merged endpoints: {api_index.get('endpoint_count', 0)}")

    warnings = api_index.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nFirst endpoints:")
    for endpoint, record in list(api_index.get("endpoints", {}).items())[:15]:
        source = record.get("source", "unknown")
        params = record.get("param_names", [])
        print(f"  - {endpoint}")
        print(f"    domain: {record.get('domain')}")
        print(f"    source: {source}")
        print(f"    params: {params}")


if __name__ == "__main__":
    enriched = build_enriched_api_index()
    print_enriched_api_summary(enriched)
    print("\nSaved to: data/api_index_enriched.json")