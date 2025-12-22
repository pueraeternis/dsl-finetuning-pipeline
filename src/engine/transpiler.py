import re
from typing import Any

from src.schema import AetherisSchema


class AuraTranspiler:
    """AuraDSL to SQL translator with identifier validation."""

    def __init__(self, schema: AetherisSchema):
        self.schema = schema

    def translate(self, dsl_query: str) -> tuple[str, list[Any]]:
        """Translates DSL to SQL returning (query_string, params)."""
        parts = [p.strip() for p in dsl_query.split("|>")]

        table_name = ""
        where_clauses: list[str] = []
        params: list[Any] = []
        select_cols = "*"
        group_by = ""

        for part in parts:
            if part.startswith("SOURCE"):
                raw_table = part.replace("SOURCE", "").strip()
                target_table = self.schema.get_table(raw_table)
                if not target_table:
                    raise ValueError(f"Table {raw_table} not in whitelist.")
                table_name = f'"{target_table.name}"'

            elif part.startswith("FILTER"):
                # Regex for: column == 'value'
                match = re.search(r"(\w+)\s*==\s*['\"](.+?)['\"]", part)
                if match:
                    col, val = match.groups()
                    table_obj = self.schema.get_table(table_name.strip('"'))
                    if table_obj and col in table_obj.column_names:
                        where_clauses.append(f'"{col}" = ?')
                        params.append(val)

            elif part.startswith("AGGREGATE"):
                # AGGREGATE SUM(kwh) BY device_id
                match = re.match(r"AGGREGATE\s+(\w+)\((\w+)\)\s+BY\s+(\w+)", part)
                if match:
                    func, col, group_col = match.groups()
                    # Validate columns
                    table_obj = self.schema.get_table(table_name.strip('"'))
                    if table_obj and all(c in table_obj.column_names for c in [col, group_col]):
                        select_cols = f'"{group_col}", {func}("{col}")'
                        group_by = f'GROUP BY "{group_col}"'

        query_parts = [f"SELECT {select_cols}", f"FROM {table_name}"]
        if where_clauses:
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
        if group_by:
            query_parts.append(group_by)

        return " ".join(query_parts), params
