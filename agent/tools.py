"""DuckDB tools exposed to the Claude agent via tool use."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb


# ---------------------------------------------------------------------------
# Safety: only allow read-only SQL
# ---------------------------------------------------------------------------
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|COPY|ATTACH|DETACH|LOAD|INSTALL)\b",
    re.IGNORECASE,
)

MAX_ROWS = 500


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


# ---------------------------------------------------------------------------
# Tool 1: execute_sql
# ---------------------------------------------------------------------------
def execute_sql(db_path: str, sql: str) -> str:
    """Execute a read-only SQL query against DuckDB and return formatted results."""
    if _FORBIDDEN_PATTERN.search(sql):
        return "ERROR: Only SELECT / WITH / EXPLAIN statements are allowed."

    con = _connect(db_path)
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(MAX_ROWS + 1)

        truncated = len(rows) > MAX_ROWS
        if truncated:
            rows = rows[:MAX_ROWS]

        # Format as a readable table string
        lines: list[str] = []
        lines.append(" | ".join(columns))
        lines.append("-+-".join("-" * max(len(c), 12) for c in columns))
        for row in rows:
            lines.append(" | ".join(_fmt(v) for v in row))

        output = "\n".join(lines)
        if truncated:
            output += f"\n\n... truncated at {MAX_ROWS} rows. Add a LIMIT clause for smaller results."
        output += f"\n\n({len(rows)} row{'s' if len(rows) != 1 else ''})"
        return output
    except Exception as e:
        return f"SQL ERROR: {e}"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Tool 2: list_tables
# ---------------------------------------------------------------------------
def list_tables(db_path: str, schema: str | None = None) -> str:
    """List tables in one or all schemas, with row counts."""
    con = _connect(db_path)
    try:
        if schema:
            tables = con.execute(
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = ?
                ORDER BY table_name
                """,
                [schema],
            ).fetchall()
        else:
            tables = con.execute(
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_schema, table_name
                """
            ).fetchall()

        if not tables:
            return f"No tables found{' in schema ' + schema if schema else ''}."

        lines: list[str] = []
        for tbl_schema, tbl_name, tbl_type in tables:
            try:
                count = con.execute(
                    f'SELECT COUNT(*) FROM "{tbl_schema}"."{tbl_name}"'
                ).fetchone()[0]
            except Exception:
                count = "?"
            kind = "VIEW" if "VIEW" in tbl_type.upper() else "TABLE"
            lines.append(f"  {tbl_schema}.{tbl_name} ({kind}, {count:,} rows)")

        return "\n".join(lines)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Tool 3: describe_table
# ---------------------------------------------------------------------------
def describe_table(db_path: str, schema: str, table: str) -> str:
    """Describe a table: columns, types, nullability, sample rows, basic stats."""
    con = _connect(db_path)
    try:
        # Column metadata
        cols = con.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()

        if not cols:
            return f"Table {schema}.{table} not found."

        lines: list[str] = [f"## {schema}.{table}\n"]
        lines.append("Columns:")
        for col_name, data_type, nullable in cols:
            null_str = ", nullable" if nullable == "YES" else ""
            lines.append(f"  - {col_name} ({data_type}{null_str})")

        # Row count
        count = con.execute(
            f'SELECT COUNT(*) FROM "{schema}"."{table}"'
        ).fetchone()[0]
        lines.append(f"\nRow count: {count:,}")

        # Sample rows
        sample = con.execute(
            f'SELECT * FROM "{schema}"."{table}" LIMIT 3'
        )
        sample_cols = [desc[0] for desc in sample.description]
        sample_rows = sample.fetchall()

        lines.append("\nSample rows (3):")
        lines.append("  " + " | ".join(sample_cols))
        for row in sample_rows:
            lines.append("  " + " | ".join(_fmt(v) for v in row))

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Tool 4: get_metadata_context
# ---------------------------------------------------------------------------
def get_metadata_context(project_dir: str, topic: str) -> str:
    """Return dbt model SQL or data lineage info based on a topic keyword.

    Examples of topic: 'fct_orders', 'staging', 'lineage', 'campaigns'
    """
    project = Path(project_dir)
    dbt_dir = project / "dbt_project" / "models"
    evidence_dir = project / "evidence" / "sources"

    # Collect all .sql and .yml files
    candidates: list[tuple[str, Path]] = []
    for d in [dbt_dir, evidence_dir]:
        if d.exists():
            for f in d.rglob("*"):
                if f.suffix in (".sql", ".yml", ".yaml"):
                    candidates.append((f.stem, f))

    # Simple keyword matching on topic
    keywords = topic.lower().split()
    matches: list[tuple[int, str, Path]] = []
    for stem, path in candidates:
        score = sum(1 for kw in keywords if kw in stem.lower() or kw in str(path).lower())
        if score > 0:
            matches.append((score, stem, path))

    # Also always include lineage info if requested
    if any(kw in ("lineage", "blood", "lineage", "pipeline", "flow") for kw in keywords):
        lineage = (
            "Data Lineage:\n"
            "  raw.products → staging.stg_products → marts.dim_products\n"
            "  raw.users → staging.stg_users → marts.dim_customers\n"
            "  raw.transactions → staging.stg_transactions ─┐\n"
            "  raw.users → staging.stg_users ────────────────┤→ marts.fct_orders\n"
            "  raw.products → staging.stg_products ──────────┤\n"
            "  raw.campaigns → staging.stg_campaigns ────────┘\n"
        )
        if not matches:
            return lineage

    matches.sort(key=lambda x: -x[0])

    if not matches:
        return f"No metadata files found matching topic '{topic}'. Try keywords like: fct_orders, staging, campaigns, lineage."

    # Return top 5 matching files' content
    output_parts: list[str] = []
    for _, stem, path in matches[:5]:
        try:
            content = path.read_text()
            rel = path.relative_to(project)
            output_parts.append(f"### {rel}\n```\n{content}\n```")
        except Exception as e:
            output_parts.append(f"### {path.name}\nError reading: {e}")

    result = "\n\n".join(output_parts)

    # Append lineage if requested
    if any(kw in ("lineage", "blood", "pipeline", "flow") for kw in keywords):
        result += "\n\n" + lineage  # noqa: F821 — lineage is defined in the branch above

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(value: object) -> str:
    """Format a single cell value for display."""
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


# ---------------------------------------------------------------------------
# Tool definitions for Claude API
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "execute_sql",
        "description": (
            "Execute a read-only SQL query against the DuckDB data warehouse. "
            "Use DuckDB SQL dialect (supports DATE_TRUNC, EXTRACT, LIST aggregates, etc). "
            "Only SELECT/WITH/EXPLAIN are allowed. Results are limited to 500 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "list_tables",
        "description": (
            "List all tables in the data warehouse, or in a specific schema. "
            "Returns table names, types (TABLE/VIEW), and row counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "Schema to list tables from (e.g. 'raw', 'staging', 'marts'). Omit to list all schemas.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "describe_table",
        "description": (
            "Get detailed metadata about a table: column names, types, nullability, "
            "row count, and 3 sample rows. Use this to understand table structure before writing SQL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "The schema name (e.g. 'raw', 'staging', 'marts').",
                },
                "table": {
                    "type": "string",
                    "description": "The table name (e.g. 'fct_orders', 'stg_users').",
                },
            },
            "required": ["schema", "table"],
        },
    },
    {
        "name": "get_metadata_context",
        "description": (
            "Retrieve dbt model SQL, data lineage, or other metadata about the data platform. "
            "Use topic keywords to search, e.g. 'fct_orders transformation', 'data lineage', "
            "'staging cleanup', 'campaigns'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Keywords to search for in dbt models and metadata files.",
                }
            },
            "required": ["topic"],
        },
    },
]
