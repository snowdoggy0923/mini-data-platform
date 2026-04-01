# Mini Data Platform

If you have an applied AI interview at Astronomer, we'll ask you to build a small project around this repo. You can also proactively do this as part of your application to speed up the process.

This repo is a synthetic data platform containing mock data csv files, Airflow DAGs, dbt models, Evidence dashboards, and a DuckDB data warehouse. Your objective is to create an agent exposed via a CLI to interact with the data platform. This CLI agent should be geared specifically towards ad-hoc questions and analysis. Things like:

- How much in sales did we do last quarter?
- Which two products are most frequently bought together?
- Are there any anomalies with how we sell products?
- What's our average customer lifetime value?
- ... and other, more complex things!

To complete this, clone the repo:

```bash
git clone https://github.com/astronomer/mini-data-platform.git
```

Then build a CLI agent where you can send questions like the ones above. We have no particular requirements around languages, model providers, methods, etc - instead, we want you to demonstrate how you think about these problems! While this repo is representative of an e-commerce company's data platform, you should aim to keep your implementation generic enough that you could plug in other "mini data platforms". See how much you can infer based on the code and warehouse metadata instead of providing explicit documentation about this data platform to the agent upfront.

To submit, share your repo with us. You should modify / create a new README that outlines your approach and where you'd continue building things if you had more time. This should take no more than a few hours.

## CLI Agent

A natural language agent that answers ad-hoc data questions by querying the DuckDB warehouse directly. Built with Claude (Anthropic API), streaming tool-use, and tiered metadata discovery inspired by [Astronomer's Kepler architecture](https://www.astronomer.io/blog/building-kepler-astronomer-internal-data-assistant/).

### Setup

```bash
uv sync
cp .env.example .env  # then add your ANTHROPIC_API_KEY
```

### Usage

```bash
# Single question
uv run python -m agent ask "How much in sales did we do last quarter?"

# Interactive chat
uv run python -m agent chat

# Point to a different database
uv run python -m agent ask "List all tables" --db path/to/other.duckdb
```

### Test Questions

```bash
# Sales analysis
uv run python -m agent ask "How much in sales did we do in Q4 2024?"
uv run python -m agent ask "What are the monthly revenue trends for 2024?"
uv run python -m agent ask "Which payment method generates the most revenue?"

# Product analysis
uv run python -m agent ask "Which two products are most frequently bought together?"
uv run python -m agent ask "What are the top 10 products by revenue?"
uv run python -m agent ask "Are there any anomalies with how we sell products?"

# Customer analysis
uv run python -m agent ask "What's our average customer lifetime value?"
uv run python -m agent ask "How do customer segments compare in terms of spending?"
uv run python -m agent ask "Which states have the most high-value customers?"

# Campaign / marketing
uv run python -m agent ask "Which campaign types have the best ROI?"
uv run python -m agent ask "How much revenue is attributed to marketing campaigns?"

# Complex / multi-step
uv run python -m agent ask "What's the year-over-year growth rate for each product category?"
uv run python -m agent ask "Find customers who haven't ordered in the last 6 months but were previously active"
```

### Evaluation

Run the evaluation suite to measure agent accuracy against ground truth SQL:

```bash
uv run python eval.py
```

This runs 5 test cases (easy → hard), checks numeric accuracy against DuckDB ground truth, and uses Claude Haiku as an LLM judge to score correctness, completeness, and insight (1-5 each).

### Architecture

The agent is inspired by [Astronomer's Kepler architecture](https://www.astronomer.io/blog/building-kepler-astronomer-internal-data-assistant/) and uses a tiered metadata discovery approach:

- **Tier 1 (marts)**: Full column names, types, and business rules injected into the system prompt — the agent can write SQL immediately without tool calls.
- **Tier 2 (raw/staging)**: Only table names and row counts in the system prompt — the agent uses `describe_table` to explore details on demand.

This keeps the prompt compact while giving the agent everything it needs for the most common analytical queries.

**Key features:**

- **Streaming tool-use loop**: Claude Sonnet with 4 tools (`execute_sql`, `list_tables`, `describe_table`, `get_metadata_context`). The agent iterates until it has enough information to answer.
- **Semantic search for table discovery**: VoyageAI embeddings (`voyage-3.5-lite`) index all table metadata at startup. At query time, the most relevant tables are surfaced by cosine similarity and injected as hints. Requires `VOYAGE_API_KEY` in `.env` (optional — agent works without it).
- **Playbooks**: Successful query patterns (question + SQL + tables used) are cached to `playbooks/playbooks.jsonl` with embeddings. On future similar questions, matched playbooks are injected into the prompt as reference patterns.
- **Context compression**: When conversation history exceeds ~50K tokens, older messages are summarized by Claude Haiku and replaced with a compact summary, keeping the most recent 8 messages intact.

---

## Quick Setup

Run the setup script to initialize everything:

```bash
./setup.sh
```

This will:
1. Generate synthetic data
2. Initialize Airflow and load data into DuckDB
3. Run dbt transformations

Then view the dashboards:

```bash
cd evidence
npm install       # First time only
npm run sources   # Build data sources
npm run dev       # Start dev server
# Open http://localhost:3000
```

---

## Manual Setup (Advanced)

<details>
<summary>Click to expand manual setup steps</summary>

### 1. Install dependencies

```bash
uv sync
```

### 2. Generate synthetic data

```bash
uv run python scripts/generate_all.py
```

### 3. Initialize Airflow

First, update `airflow/airflow.cfg` to use an absolute path for the database:

```bash
cd airflow
# Update sql_alchemy_conn in airflow.cfg to:
# sql_alchemy_conn = sqlite:////absolute/path/to/your/mini-data-platform/airflow/airflow.db

export AIRFLOW_HOME=$(pwd)
uv run airflow db migrate
```

### 4. Run ingestion DAGs

```bash
# From airflow/ directory
export AIRFLOW_HOME=$(pwd)
uv run python dags/ingest_products.py
uv run python dags/ingest_users.py
uv run python dags/ingest_transactions.py
uv run python dags/ingest_campaigns.py
uv run python dags/ingest_pageviews.py
```

### 5. Run dbt transformations

```bash
# From airflow/ directory
export AIRFLOW_HOME=$(pwd)
uv run python dags/run_dbt.py

# Or run dbt directly
cd ../dbt_project
uv run dbt build --profiles-dir .
```

</details>

## Project Structure

```sh
mini-data-platform/
├── sources/              # Raw source data (CSV files)
│   ├── postgres/         # Sales, products, users
│   ├── salesforce/       # Marketing campaigns
│   └── analytics/        # Page view events
├── airflow/
│   ├── dags/            # Airflow DAGs for ingestion and transformation
│   │   ├── ingest_*.py  # Load data from sources → raw schema
│   │   ├── run_dbt.py   # Run dbt staging → marts pipeline
│   │   └── build_evidence.py  # Build Evidence dashboards
│   └── utils/           # Shared utilities
├── warehouse/           # DuckDB database (data.duckdb)
├── dbt_project/         # dbt transformations
│   └── models/
│       ├── staging/     # Clean raw data (5 models)
│       └── marts/       # Analytics-ready tables (3 models)
├── evidence/            # Evidence BI dashboards
│   ├── pages/           # Dashboard pages (index, sales, products, customers)
│   └── sources/         # SQL queries and connection
└── scripts/             # Data generation scripts
```

## Data Pipeline

### Raw Layer (`raw` schema)

- Loaded by Airflow ingestion DAGs
- 5 tables: products, users, transactions, campaigns, pageviews

### Staging Layer (`staging` schema)

- Created by dbt
- 5 views: stg_products, stg_users, stg_transactions, stg_campaigns, stg_pageviews

### Marts Layer (`marts` schema)

- Created by dbt
- Denormalized tables for analysis
- 3 tables:
  - `dim_products`: Current product catalog (62 products)
  - `dim_customers`: Current customer info (5,000 customers)
  - `fct_orders`: Order line items with dimensions (35,980 rows)

## Data Volumes

- **Raw**: ~93K total rows across 5 tables
- **Staging**: Same as raw (views)
- **Marts**: 5,062 dimension rows + 35,980 fact rows
- **Database Size**: ~5-10 MB (DuckDB)

## Evidence Dashboards

The project includes interactive dashboards built with Evidence:

### Available Dashboards

1. **Overview** (`/`) - Key metrics, revenue trends, category performance
2. **Sales** (`/sales`) - Daily/monthly sales, country analysis, recent orders
3. **Products** (`/products`) - Product performance, category trends, price analysis
4. **Customers** (`/customers`) - Customer segments, lifetime value, acquisition trends

### Running Evidence

```bash
cd evidence
npm install       # First time only
npm run sources   # Build data sources
npm run dev       # Start dev server
```

Then open http://localhost:3000 to view dashboards.

**Note**: Evidence connects to the DuckDB warehouse at `../warehouse/data.duckdb` and queries the `marts` schema through pass-through SQL files (`fct_orders.sql`, `dim_customers.sql`, `dim_products.sql`).

### Building Evidence (Static Site)

```bash
# Using Airflow DAG
cd airflow
uv run python dags/build_evidence.py

# Or build directly
cd evidence
npm run build
```
