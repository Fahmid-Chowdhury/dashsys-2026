from src.schema_index import build_schema_index, save_schema_index


def main():
    schema_index = build_schema_index(
        snapshot_dir="DBSnapshot",
        include_samples=False,
    )

    print("Table count:", schema_index["table_count"])
    print("Tables found:")

    for table_name in schema_index["tables"]:
        table = schema_index["tables"][table_name]
        print(
            f"- {table_name}: "
            f"{len(table['columns'])} columns, "
            f"{table['row_count']} rows, "
            f"kind={table['kind']}, "
            f"domain={table['domain']}"
        )

    print("\nJoin candidates:")
    for join in schema_index["join_candidates"][:10]:
        print(
            f"- {join['left_table']}.{join['left_column']} "
            f"= {join['right_table']}.{join['right_column']} "
            f"({join['confidence']})"
        )

    save_schema_index(schema_index, "data/schema_index_test.json")
    print("\nSaved test index to: data/schema_index_test.json")


if __name__ == "__main__":
    main()