import random
from typing import Any

from src.schema import ColumnSchema, ColumnType


class SkeletonGenerator:
    """Generates DSL skeletons with placeholders for semantic infilling."""

    def __init__(self, schema: Any):
        self.schema = schema
        # Mapping column names to generic placeholder types
        self.placeholder_map = {
            "room": "{{ROOM_NAME}}",
            "device_id": "{{DEVICE_ID}}",
            "name": "{{DEVICE_NAME}}",
            "type": "{{DEVICE_TYPE}}",
            "manufacturer": "{{MANUFACTURER}}",
            "status": "{{STATUS_VALUE}}",
            "event_type": "{{EVENT_TYPE}}",
            "severity": "{{SEVERITY_LEVEL}}",
            "authorized_person": "{{PERSON_NAME}}",
            "sensor_id": "{{SENSOR_ID}}",
            "activity_level": "{{ACTIVITY_LEVEL}}",
            "mode": "{{MODE_NAME}}",
            "command_text": "{{COMMAND_TEXT}}",
        }

    def _get_value_skeleton(self, col: ColumnSchema) -> str:
        """Returns either a numeric value or a placeholder string."""
        if col.type in [ColumnType.INTEGER, ColumnType.REAL]:
            low = col.min_val if col.min_val is not None else 0
            high = col.max_val if col.max_val is not None else 100
            if col.type == ColumnType.INTEGER:
                return str(random.randint(int(low), int(high)))
            return str(round(random.uniform(low, high), 2))

        if col.type == ColumnType.DATETIME:
            return "{{TIMESTAMP}}"

        # Default to placeholder from map or generic placeholder
        return self.placeholder_map.get(col.name, f"{{{{VALUE_{col.name.upper()}}}}}")

    def generate_skeleton(self) -> dict[str, Any]:
        """Creates a single DSL skeleton with technical metadata."""
        table = random.choice(self.schema.tables)
        complexity = random.choices(
            ["simple", "filter", "agg", "full"],
            weights=[15, 30, 35, 20],
        )[0]

        dsl_parts = [f"SOURCE {table.name}"]
        used_cols = set()

        # 1. FILTER
        if complexity in ["filter", "agg", "full"]:
            col = random.choice(table.columns)
            val = self._get_random_formatted_val(col)
            dsl_parts.append(f"FILTER {col.name} == {val}")
            used_cols.add(col.name)

        # 2. AGGREGATE
        if complexity in ["agg", "full"]:
            num_cols = [c for c in table.columns if c.type in [ColumnType.REAL, ColumnType.INTEGER]]
            # Text columns for grouping
            grp_cols = [c for c in table.columns if c.type == ColumnType.TEXT and c.name not in used_cols]

            if num_cols and grp_cols:
                n_col = random.choice(num_cols)
                g_col = random.choice(grp_cols)
                func = random.choice(["SUM", "AVG", "MAX", "MIN", "COUNT"])
                dsl_parts.append(f"AGGREGATE {func}({n_col.name}) BY {g_col.name}")
                used_cols.add(n_col.name)
                used_cols.add(g_col.name)

        # 3. SORT & LIMIT
        if complexity == "full":
            sort_col = random.choice(table.columns)
            dsl_parts.append(f"SORT {sort_col.name} DESC")
            dsl_parts.append("LIMIT 5")

        return {
            "table_name": table.name,
            "table_description": table.description,
            "columns_info": [{"name": c.name, "description": c.description} for c in table.columns],
            "dsl_skeleton": " |> ".join(dsl_parts),
        }

    def _get_random_formatted_val(self, col: ColumnSchema) -> str:
        """Helper to wrap text placeholders in quotes and leave numbers raw."""
        val = self._get_value_skeleton(col)
        if col.type in [ColumnType.TEXT, ColumnType.DATETIME]:
            return f"'{val}'"
        return val
