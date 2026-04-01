"""Evaluation script for the data platform agent.

Runs test cases against the agent and verifies:
1. Numeric accuracy — does the response contain the correct numbers?
2. Response quality — LLM-as-judge scoring (correctness, completeness, insight)
"""
from __future__ import annotations

import json
import re
import sys

import anthropic
import duckdb
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

DB_PATH = "warehouse/data.duckdb"
PROJECT_DIR = "."
JUDGE_MODEL = "claude-haiku-4-5-20251001"

console = Console()

# ---------------------------------------------------------------------------
# Test cases: question + ground truth SQL
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- Easy: direct aggregation ---
    {
        "question": "What is the total revenue across all orders?",
        "ground_truth_sql": "SELECT ROUND(SUM(total), 2) FROM marts.fct_orders",
    },
    {
        "question": "How many unique customers have placed orders?",
        "ground_truth_sql": "SELECT COUNT(DISTINCT user_id) FROM marts.fct_orders",
    },
    # --- Medium: filtering + grouping ---
    {
        "question": "How many distinct product categories do we have?",
        "ground_truth_sql": "SELECT COUNT(DISTINCT category) FROM marts.dim_products WHERE category IS NOT NULL",
    },
    {
        "question": "What is the top selling product by total revenue?",
        "ground_truth_sql": (
            "SELECT product_name, ROUND(SUM(total), 2) as rev "
            "FROM marts.fct_orders WHERE status != 'cancelled' "
            "GROUP BY product_name ORDER BY rev DESC LIMIT 1"
        ),
    },
    # --- Hard: time-based + multi-step ---
    {
        "question": "How much revenue did we make in Q4 2024?",
        "ground_truth_sql": (
            "SELECT ROUND(SUM(total), 2) FROM marts.fct_orders "
            "WHERE transaction_date >= '2024-10-01' AND transaction_date < '2025-01-01' "
            "AND status != 'cancelled'"
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ground_truth(sql: str) -> list:
    """Execute ground truth SQL and return the result row."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql).fetchone()
    finally:
        con.close()


def check_number_in_response(response: str, value) -> bool:
    """Check if a numeric value appears in the response in any common format."""
    if value is None:
        return False

    if isinstance(value, str):
        # For string values (like product names), check substring
        return value.lower() in response.lower()

    num = float(value)
    formats_to_check = []

    # Exact with 2 decimals: 36204723.67
    formats_to_check.append(f"{num:.2f}")
    # With commas: 36,204,723.67
    formats_to_check.append(f"{num:,.2f}")
    # Integer form: 36204724
    formats_to_check.append(str(int(round(num))))
    # Integer with commas: 36,204,724
    formats_to_check.append(f"{int(round(num)):,}")

    # Millions: 36.2M or 36.20M
    if abs(num) >= 1_000_000:
        m = num / 1_000_000
        formats_to_check.append(f"{m:.1f}M")
        formats_to_check.append(f"{m:.2f}M")
        formats_to_check.append(f"${m:.1f}M")
        formats_to_check.append(f"${m:.2f}M")
        formats_to_check.append(f"{m:.1f} million")

    # Thousands: 36,205K
    if abs(num) >= 1_000:
        k = num / 1_000
        formats_to_check.append(f"{k:.1f}K")

    # With dollar sign
    formats_to_check.append(f"${num:,.2f}")
    formats_to_check.append(f"${int(round(num)):,}")

    # Remove whitespace in response for matching
    normalized = response.replace(" ", "").replace("\n", "")

    for fmt in formats_to_check:
        if fmt.replace(" ", "") in normalized:
            return True

    return False


def llm_judge(question: str, ground_truth: str, response: str) -> dict:
    """Use Claude as a judge to score the response quality."""
    client = anthropic.Anthropic()
    prompt = f"""You are evaluating a data analysis agent's response. Score it on three dimensions (1-5 each).

Question asked: "{question}"
Ground truth data: {ground_truth}
Agent's response:
---
{response[:3000]}
---

Score each dimension and provide a brief justification. Respond in JSON format:
{{
  "correctness": {{"score": <1-5>, "reason": "<brief reason>"}},
  "completeness": {{"score": <1-5>, "reason": "<brief reason>"}},
  "insight": {{"score": <1-5>, "reason": "<brief reason>"}}
}}

Scoring guide:
- correctness: Are the numbers accurate? Does the response make false claims?
- completeness: Does it fully answer what was asked?
- insight: Does it provide useful analysis beyond just numbers?"""

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text

    # Extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group())
    return {"correctness": {"score": 0}, "completeness": {"score": 0}, "insight": {"score": 0}}


def run_agent_question(question: str) -> str:
    """Run a question through the agent and return the text response."""
    from agent.agent import DataAgent

    agent = DataAgent(db_path=DB_PATH, project_dir=PROJECT_DIR)
    return agent.chat(question)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Panel("[bold]Agent Evaluation[/bold]", border_style="blue"))

    results: list[dict] = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        console.print(f"\n[bold cyan]Test {i}/{len(TEST_CASES)}:[/bold cyan] {question}")

        # Get ground truth
        gt_row = get_ground_truth(tc["ground_truth_sql"])
        gt_display = " | ".join(str(v) for v in gt_row)
        console.print(f"  Ground truth: [green]{gt_display}[/green]")

        # Run agent
        console.print("  Running agent...", end="")
        try:
            response = run_agent_question(question)
            console.print(" [green]done[/green]")
        except Exception as e:
            console.print(f" [red]ERROR: {e}[/red]")
            results.append({"question": question, "numeric_match": False, "scores": {}})
            continue

        # Check numeric accuracy
        numeric_match = any(check_number_in_response(response, v) for v in gt_row)
        match_str = "[green]PASS[/green]" if numeric_match else "[red]FAIL[/red]"
        console.print(f"  Numeric match: {match_str}")

        # LLM judge
        console.print("  Judging quality...", end="")
        try:
            scores = llm_judge(question, gt_display, response)
            console.print(" [green]done[/green]")
            for dim in ("correctness", "completeness", "insight"):
                s = scores.get(dim, {})
                score = s.get("score", "?")
                reason = s.get("reason", "")
                console.print(f"    {dim.capitalize():15s} {score}/5  {reason}")
        except Exception as e:
            console.print(f" [red]ERROR: {e}[/red]")
            scores = {}

        results.append({
            "question": question,
            "numeric_match": numeric_match,
            "scores": scores,
        })

    # ---- Summary ----
    console.print("\n")
    table = Table(title="Evaluation Summary", border_style="blue")
    table.add_column("#", width=3)
    table.add_column("Question", max_width=50)
    table.add_column("Numeric", justify="center", width=9)
    table.add_column("Correct", justify="center", width=9)
    table.add_column("Complete", justify="center", width=9)
    table.add_column("Insight", justify="center", width=9)

    total_numeric = 0
    total_scores = {"correctness": [], "completeness": [], "insight": []}

    for i, r in enumerate(results, 1):
        nm = r["numeric_match"]
        total_numeric += int(nm)
        nm_str = "[green]PASS[/green]" if nm else "[red]FAIL[/red]"

        score_strs = []
        for dim in ("correctness", "completeness", "insight"):
            s = r["scores"].get(dim, {}).get("score", "?")
            if isinstance(s, (int, float)):
                total_scores[dim].append(s)
            score_strs.append(str(s))

        table.add_row(str(i), r["question"], nm_str, *score_strs)

    console.print(table)

    # Averages
    n = len(results)
    console.print(f"\n  Numeric accuracy: {total_numeric}/{n} ({100*total_numeric/n:.0f}%)")
    for dim in ("correctness", "completeness", "insight"):
        vals = total_scores[dim]
        avg = sum(vals) / len(vals) if vals else 0
        console.print(f"  Avg {dim.capitalize():15s} {avg:.1f}/5")

    # Exit code
    all_pass = total_numeric == n and all(
        r["scores"].get("correctness", {}).get("score", 0) >= 3 for r in results
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
