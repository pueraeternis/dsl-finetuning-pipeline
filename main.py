import sys

from src.engine.db import DBManager
from src.engine.transpiler import AuraTranspiler
from src.retrieval.vector_store import SchemaRetriever
from src.schema import AETHERIS_DB


def main() -> None:
    """Main entry point for testing the pipeline foundation."""
    # Initialize Components
    db_manager = DBManager(schema=AETHERIS_DB)
    transpiler = AuraTranspiler(schema=AETHERIS_DB)

    # Setup Database
    try:
        db_manager.setup_db()
        print("[OK] Database initialized.")
    except Exception as e:
        print(f"[ERR] DB Setup failed: {e}")
        sys.exit(1)

    # Test Transpilation
    test_dsl = "SOURCE energy_consumption |> FILTER device_id == 'fridge_01' |> AGGREGATE SUM(kwh) BY device_id"

    try:
        sql, params = transpiler.translate(test_dsl)
        print(f"\nDSL: {test_dsl}")
        print(f"SQL: {sql}")
        print(f"PARAMS: {params}")

        # Try executing (will return empty list as we haven't seeded data yet)
        results = db_manager.execute_query(sql, params)
        print(f"Results: {results}")

    except ValueError as e:
        print(f"[ERR] Validation failed: {e}")

    # Initialize Retriever
    retriever = SchemaRetriever(schema=AETHERIS_DB)

    # Test cases for RAG
    test_queries = [
        "How much electricity did I use yesterday?",
        "Is there anyone in the living room right now?",
        "Check if the front door is locked or any security alerts.",
    ]

    print("\n--- RAG TEST ---")
    for query in test_queries:
        tables = retriever.get_relevant_tables(query, top_k=1)
        found_table = tables[0].name if tables else "None"
        print(f"NL Query: '{query}'")
        print(f"Matched Table: {found_table}\n")


if __name__ == "__main__":
    main()
