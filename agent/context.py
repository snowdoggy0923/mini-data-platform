"""Build the system prompt by auto-discovering warehouse metadata at startup."""
from __future__ import annotations

from pathlib import Path

import duckdb


def build_system_prompt(db_path: str, project_dir: str) -> str:
    """Scan DuckDB metadata + dbt files and return a tiered system prompt.

    Tier 1 (marts): full column info + business rules — Agent can write SQL immediately.
    Tier 2 (raw/staging): table name + row count overview — Agent explores via tools when needed.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        sections: list[str] = [_PREAMBLE]

        # ---- Tier 1: marts tables (full detail) ----
        marts_tables = _get_tables(con, "marts")
        if marts_tables:
            sections.append("## Key Tables (marts layer — preferred for analysis, cleaned data)\n")
            for tbl_name, row_count in marts_tables:
                cols = _get_columns(con, "marts", tbl_name)
                col_list = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
                sections.append(
                    f"### marts.{tbl_name} ({row_count:,} rows)\n"
                    f"Columns: {col_list}\n"
                )

            # Business rules inferred from dbt models
            sections.append(_get_business_rules(project_dir))

        # ---- Tier 2: other schemas (overview only) ----
        other_schemas: list[str] = []
        for schema in _get_schemas(con):
            if schema in ("marts", "information_schema", "pg_catalog"):
                continue
            tables = _get_tables(con, schema)
            if tables:
                table_list = ", ".join(f"{name} ({count:,})" for name, count in tables)
                other_schemas.append(f"- **{schema}** layer: {table_list}")

        if other_schemas:
            sections.append(
                "## Other Available Tables (use describe_table tool to explore details)\n"
                + "\n".join(other_schemas)
            )

        # ---- Data lineage + time range ----
        sections.append(_LINEAGE)

        # ---- Working rules ----
        sections.append(_RULES)

        return "\n\n".join(sections)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_schemas(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT table_schema FROM information_schema.tables ORDER BY table_schema"
    ).fetchall()
    return [r[0] for r in rows]


def _get_tables(con: duckdb.DuckDBPyConnection, schema: str) -> list[tuple[str, int]]:
    tables = con.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = ? ORDER BY table_name
        """,
        [schema],
    ).fetchall()
    result = []
    for (tbl_name,) in tables:
        try:
            count = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{tbl_name}"').fetchone()[0]
        except Exception:
            count = 0
        result.append((tbl_name, count))
    return result


def _get_columns(
    con: duckdb.DuckDBPyConnection, schema: str, table: str
) -> list[tuple[str, str]]:
    return con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema, table],
    ).fetchall()


def _get_business_rules(project_dir: str) -> str:
    """Extract key business rules from dbt model SQL files."""
    rules: list[str] = ["**Business Rules (inferred from dbt models):**"]

    dbt_dir = Path(project_dir) / "dbt_project" / "models"
    if not dbt_dir.exists():
        return ""

    # Read fct_orders.sql to extract key info
    fct_path = dbt_dir / "marts" / "fct_orders.sql"
    if fct_path.exists():
        content = fct_path.read_text()
        if "transaction_id" in content:
            rules.append(
                "- `fct_orders`: A single transaction_id can have multiple rows (multi-product orders). "
                "Deduplicate by transaction_id when aggregating order-level metrics."
            )
        if "line_margin" in content:
            rules.append("- `fct_orders.line_margin` = total - (product_cost × quantity)")
        if "is_discounted" in content:
            rules.append("- `fct_orders.is_discounted` = discount > 0")

    # Read staging models for cleanup rules
    stg_dir = dbt_dir / "staging"
    if stg_dir.exists():
        for f in stg_dir.glob("*.sql"):
            content = f.read_text()
            if "abs(" in content.lower():
                rules.append(f"- `{f.stem}`: Negative values converted to absolute values (data cleaning)")

    return "\n".join(rules)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are a data analysis assistant connected to an e-commerce company's DuckDB data warehouse.
Answer questions by exploring the warehouse metadata and writing SQL queries.
Always respond in the same language as the user's question."""

_LINEAGE = """\
## Data Lineage
```
raw.* → staging.stg_* (dbt views, data cleaning) → marts.* (dbt tables, dimensional modeling)
```
- raw → staging: fixes negative values, filters nulls, caps impossible metrics
- staging → marts: denormalized joins, calculated fields (margin, is_discounted)
- Data time range: 2020-01-01 to 2024-12-31"""

_RULES = """\
## Working Rules
1. Prefer marts tables — they are clean, tested, and denormalized for analysis.
2. Before writing SQL, think about whether you need to explore table structure first.
3. For fct_orders aggregations, remember transaction_id has a one-to-many relationship with line items.
4. Use DuckDB SQL dialect (DATE_TRUNC, EXTRACT, LIST aggregate, PIVOT, etc).
5. Include concrete numbers in your answers. Show the SQL query for transparency.
6. If you need to understand data transformation logic, use the get_metadata_context tool.
7. When uncertain about column meaning, use describe_table to check sample data first."""
