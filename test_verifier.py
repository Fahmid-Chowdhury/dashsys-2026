from src.verifier import (
    load_api_index_if_available,
    load_json,
    validate_api_call,
    validate_sql_call,
    validate_tool_call,
)


def print_result(title, result):
    print("\n" + title)
    print("=" * len(title))
    print(result.to_dict())


def main():
    metadata = load_json("data/sample_metadata.json")
    api_index = load_api_index_if_available("data/api_index_enriched.json")

    good_sql = """
    SELECT CAMPAIGN.NAME AS CAMPAIGNNAME,
           CAMPAIGN.CAMPAIGNID
    FROM DIM_CAMPAIGN AS CAMPAIGN
    LIMIT 10
    """

    bad_sql_table = """
    SELECT J.NAME
    FROM DIM_JOURNEY AS J
    LIMIT 10
    """

    bad_sql_column = """
    SELECT CAMPAIGN.FAKECOLUMN
    FROM DIM_CAMPAIGN AS CAMPAIGN
    LIMIT 10
    """
    
    bad_sql_unqualified_column = """
    SELECT FAKECOLUMN
    FROM DIM_CAMPAIGN
    LIMIT 10
    """

    good_api = {
        "method": "GET",
        "url": "/ajo/journey",
        "params": {"pageSize": "10"},
    }

    bad_api_endpoint = {
        "method": "GET",
        "url": "/unifiedtags/tags",
        "params": {"limit": "10"},
    }

    bad_api_placeholder = {
        "method": "GET",
        "url": "/data/foundation/schemaregistry/tenant/schemas/{schema_id}",
        "params": {},
    }
    
    bad_api_param_placeholder = {
        "method": "GET",
        "url": "/ajo/journey",
        "params": {"filter": "name==<journey_name>"},
    }

    print_result(
        "GOOD SQL",
        validate_sql_call(good_sql, metadata),
    )

    print_result(
        "BAD SQL TABLE",
        validate_sql_call(bad_sql_table, metadata),
    )

    print_result(
        "BAD SQL COLUMN",
        validate_sql_call(bad_sql_column, metadata),
    )
    
    print_result(
        "BAD SQL UNQUALIFIED COLUMN",
        validate_sql_call(bad_sql_unqualified_column, metadata),
    )

    print_result(
        "GOOD API",
        validate_api_call(
            method=good_api["method"],
            url=good_api["url"],
            params=good_api["params"],
            metadata=metadata,
            api_index=api_index,
        ),
    )

    print_result(
        "BAD API ENDPOINT",
        validate_api_call(
            method=bad_api_endpoint["method"],
            url=bad_api_endpoint["url"],
            params=bad_api_endpoint["params"],
            metadata=metadata,
            api_index=api_index,
        ),
    )

    print_result(
        "BAD API PLACEHOLDER",
        validate_api_call(
            method=bad_api_placeholder["method"],
            url=bad_api_placeholder["url"],
            params=bad_api_placeholder["params"],
            metadata=metadata,
            api_index=api_index,
        ),
    )
    
    print_result(
        "BAD API PARAM PLACEHOLDER",
        validate_api_call(
            method=bad_api_param_placeholder["method"],
            url=bad_api_param_placeholder["url"],
            params=bad_api_param_placeholder["params"],
            metadata=metadata,
            api_index=api_index,
        ),
    )

    print_result(
        "GENERIC TOOL CALL TEST",
        validate_tool_call(
            {
                "action": "api_call",
                "method": "GET",
                "url": "/ajo/journey",
                "params": {"pageSize": "10"},
            },
            metadata=metadata,
            api_index=api_index,
        ),
    )


if __name__ == "__main__":
    main()