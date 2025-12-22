import sys

from src.engine.db import DBManager
from src.engine.transpiler import AuraTranspiler
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


if __name__ == "__main__":
    main()
