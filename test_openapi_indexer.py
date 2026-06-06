from src.openapi_indexer import (
    build_enriched_api_index,
    extract_openapi_endpoints,
    print_enriched_api_summary,
)


def test_small_openapi_doc():
    fake_doc = {
        "openapi": "3.0.0",
        "paths": {
            "/data/foundation/catalog/batches": {
                "get": {
                    "summary": "List batches",
                    "operationId": "listBatches",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer"},
                        },
                        {
                            "name": "status",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {"description": "OK"}
                    },
                }
            },
            "/data/foundation/schemaregistry/tenant/schemas/{schema_id}": {
                "get": {
                    "summary": "Get schema",
                    "operationId": "getSchema",
                    "parameters": [
                        {
                            "name": "schema_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "OK"},
                        "404": {"description": "Not found"},
                    },
                }
            },
        },
    }

    endpoints = extract_openapi_endpoints(fake_doc, source_file="fake.yaml")

    assert "GET /data/foundation/catalog/batches" in endpoints
    assert (
        "GET /data/foundation/schemaregistry/tenant/schemas/{schema_id}"
        in endpoints
    )

    batch = endpoints["GET /data/foundation/catalog/batches"]
    assert batch["query_param_names"] == ["limit", "status"]

    schema = endpoints["GET /data/foundation/schemaregistry/tenant/schemas/{schema_id}"]
    assert schema["path_param_names"] == ["schema_id"]
    assert schema["required_params"] == ["schema_id"]

    print("Small OpenAPI parser test passed.")


def main():
    test_small_openapi_doc()

    enriched = build_enriched_api_index(
        example_api_index_path="data/api_index.json",
        openapi_spec_dir="openapi_specs",
        output_path="data/api_index_enriched_test.json",
    )

    print()
    print_enriched_api_summary(enriched)
    print("\nSaved test enriched index to: data/api_index_enriched_test.json")


if __name__ == "__main__":
    main()