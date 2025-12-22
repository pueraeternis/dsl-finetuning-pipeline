import pathlib
import sqlite3
from typing import Any

from src.schema import AetherisSchema


class DBManager:
    """Manages SQLite database operations based on Pydantic schema."""

    def __init__(self, schema: AetherisSchema, db_path: str = "data/aetheris.db"):
        self.schema = schema
        self.db_path = db_path
        pathlib.Path("data").mkdir(exist_ok=True)

    def setup_db(self) -> None:
        """Initializes tables using schema definitions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for table in self.schema.tables:
                cols = ", ".join([f'"{c.name}" {c.type.value}' for c in table.columns])
                query = f'CREATE TABLE IF NOT EXISTS "{table.name}" ({cols})'
                cursor.execute(query)
            conn.commit()

    def execute_query(self, sql: str, params: list[Any]) -> list[Any]:
        """Safely executes a parameterized SQL query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()
