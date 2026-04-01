"""Claude tool-use agent loop with streaming output."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic

from agent.context import build_system_prompt
from agent.playbooks import PlaybookStore
from agent.search import TableIndex
from agent.tools import (
    TOOL_DEFINITIONS,
    describe_table,
    execute_sql,
    get_metadata_context,
    list_tables,
)

MODEL = "claude-sonnet-4-6"
SUMMARIZE_MODEL = "claude-haiku-4-5-20251001"

# Context compression thresholds
# Estimate: 1 token ≈ 4 chars. Compress when exceeding ~50K tokens.
COMPRESS_CHAR_THRESHOLD = 200_000
# Keep the most recent N messages intact (preserve current conversation context)
KEEP_RECENT_MESSAGES = 8  # ~4 turns (user + assistant pairs)


@dataclass
class ToolCall:
    """Represents a tool invocation the agent wants to make."""
    id: str
    name: str
    input: dict


@dataclass
class DataAgent:
    """A conversational data analysis agent powered by Claude with tool use."""

    db_path: str
    project_dir: str
    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    _client: anthropic.Anthropic | None = field(default=None, repr=False)
    _table_index: TableIndex | None = field(default=None, repr=False)
    _playbook_store: PlaybookStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic()
        self.system_prompt = build_system_prompt(self.db_path, self.project_dir)
        self._table_index = _build_table_index(self.db_path)
        self._playbook_store = PlaybookStore(
            str(Path(self.project_dir) / "playbooks")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, str], None] | None = None,
    ) -> str:
        """Send a user message and return the full assistant text response.

        Callbacks:
            on_text(delta):       called for each streamed text chunk
            on_tool_start(name, input):  called when a tool invocation begins
            on_tool_end(name, result):   called when a tool returns
        """
        # --- Semantic search: find relevant tables ---
        enriched_message = user_message
        try:
            if self._table_index:
                relevant = self._table_index.search(user_message, top_k=3)
                if relevant:
                    tables_hint = ", ".join(f"{t} ({s:.0%})" for t, s in relevant)
                    enriched_message += f"\n\n[Semantic search suggests these tables are most relevant: {tables_hint}]"
        except Exception:
            pass  # graceful fallback — agent works fine without semantic hints

        # --- Playbooks: inject similar past query patterns ---
        try:
            if self._playbook_store:
                similar = self._playbook_store.find_similar(user_message)
                if similar:
                    enriched_message += "\n\n" + self._playbook_store.format_for_prompt(similar)
        except Exception:
            pass  # graceful fallback

        self.messages.append({"role": "user", "content": enriched_message})
        self._maybe_compress()

        full_text = ""

        while True:
            text_chunk, tool_calls, stop_reason = self._stream_response(on_text)
            full_text += text_chunk

            if stop_reason != "tool_use" or not tool_calls:
                break

            # Execute tools and feed results back
            assistant_content: list[dict] = []
            if text_chunk:
                assistant_content.append({"type": "text", "text": text_chunk})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })

            self.messages.append({"role": "assistant", "content": assistant_content})

            # Build tool results
            tool_results: list[dict] = []
            for tc in tool_calls:
                if on_tool_start:
                    on_tool_start(tc.name, tc.input)

                result = self._execute_tool(tc.name, tc.input)

                if on_tool_end:
                    on_tool_end(tc.name, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            self.messages.append({"role": "user", "content": tool_results})
            full_text = ""  # reset for next iteration's text

        # --- Save successful query as a playbook ---
        try:
            self._save_playbook(user_message)
        except Exception:
            pass  # non-critical — don't break the response if playbook save fails

        return full_text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stream_response(
        self,
        on_text: Callable[[str], None] | None,
    ) -> tuple[str, list[ToolCall], str]:
        """Make a streaming API call. Returns (text, tool_calls, stop_reason)."""
        collected_text = ""
        tool_calls: list[ToolCall] = []

        with self._client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=self.system_prompt,
            messages=self.messages,
            tools=TOOL_DEFINITIONS,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                id=event.content_block.id,
                                name=event.content_block.name,
                                input={},
                            )
                        )
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        collected_text += event.delta.text
                        if on_text:
                            on_text(event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        # Accumulate tool input JSON
                        if tool_calls:
                            tc = tool_calls[-1]
                            tc.input = {}  # will be set from final message

            # Get the final message to extract complete tool inputs
            final = stream.get_final_message()
            stop_reason = final.stop_reason

            # Extract complete tool inputs from final message
            tc_index = 0
            for block in final.content:
                if block.type == "tool_use":
                    if tc_index < len(tool_calls):
                        tool_calls[tc_index].input = block.input if isinstance(block.input, dict) else json.loads(block.input)
                        tc_index += 1

        return collected_text, tool_calls, stop_reason

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        """Dispatch a tool call to the appropriate function."""
        if name == "execute_sql":
            return execute_sql(self.db_path, tool_input["sql"])
        elif name == "list_tables":
            return list_tables(self.db_path, tool_input.get("schema"))
        elif name == "describe_table":
            return describe_table(self.db_path, tool_input["schema"], tool_input["table"])
        elif name == "get_metadata_context":
            return get_metadata_context(self.project_dir, tool_input["topic"])
        else:
            return f"Unknown tool: {name}"

    # ------------------------------------------------------------------
    # Playbook saving
    # ------------------------------------------------------------------

    def _save_playbook(self, question: str) -> None:
        """Extract SQL queries from the conversation and save as a playbook."""
        if not self._playbook_store:
            return

        sql_queries: list[str] = []
        tables_used: set[str] = set()

        for msg in self.messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "execute_sql":
                    sql = block.get("input", {}).get("sql", "")
                    if sql:
                        sql_queries.append(sql)
                        # Extract table references from SQL
                        for match in re.findall(r"(?:FROM|JOIN)\s+([\w.]+)", sql, re.IGNORECASE):
                            tables_used.add(match)

        if sql_queries:
            self._playbook_store.save(question, sql_queries, sorted(tables_used))

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    def _estimate_chars(self) -> int:
        """Rough character count of all messages (proxy for token count)."""
        return sum(len(json.dumps(m, default=str)) for m in self.messages)

    def _maybe_compress(self) -> None:
        """Compress old messages into a summary when context grows too large.

        Strategy (sliding window + summary):
        1. Keep the most recent KEEP_RECENT_MESSAGES messages intact
        2. Summarize everything older using a cheap, fast model (Haiku)
        3. Replace old messages with the summary

        This mirrors the approach used by Claude Code itself.
        The biggest token consumers are tool_result blocks (SQL output),
        so compression is very effective.
        """
        if self._estimate_chars() < COMPRESS_CHAR_THRESHOLD:
            return

        if len(self.messages) <= KEEP_RECENT_MESSAGES:
            return

        old_messages = self.messages[:-KEEP_RECENT_MESSAGES]
        recent_messages = self.messages[-KEEP_RECENT_MESSAGES:]

        summary = self._summarize(old_messages)

        # Replace old messages with a compact summary
        # Must maintain valid message structure: user → assistant alternation
        self.messages = [
            {"role": "user", "content": f"[Previous conversation summary]:\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation and will use it to inform my answers."},
            *recent_messages,
        ]

    def _summarize(self, messages: list[dict]) -> str:
        """Use a cheap model to summarize old conversation messages."""
        # Extract readable text from messages (flatten tool_use/tool_result blocks)
        conversation_text = self._flatten_messages(messages)

        response = self._client.messages.create(
            model=SUMMARIZE_MODEL,
            max_tokens=1024,
            system=(
                "You are a conversation summarizer. Produce a concise summary of the "
                "data analysis conversation below. Focus on:\n"
                "- What questions the user asked\n"
                "- What SQL queries were run and their key results (specific numbers)\n"
                "- Any important findings or conclusions\n"
                "Keep it factual and compact. Do NOT include raw SQL output rows."
            ),
            messages=[{"role": "user", "content": conversation_text}],
        )
        return response.content[0].text

    @staticmethod
    def _flatten_messages(messages: list[dict]) -> str:
        """Convert messages (including tool blocks) into readable plain text."""
        parts: list[str] = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]

            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(f"{role}: {block['text']}")
                        elif block.get("type") == "tool_use":
                            parts.append(f"{role} [tool call]: {block['name']}({json.dumps(block.get('input', {}), default=str)[:200]})")
                        elif block.get("type") == "tool_result":
                            # Truncate large tool results — these are the main token consumers
                            result_text = str(block.get("content", ""))
                            if len(result_text) > 500:
                                result_text = result_text[:500] + "... [truncated]"
                            parts.append(f"{role} [tool result]: {result_text}")

        return "\n".join(parts)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_table_index(db_path: str) -> TableIndex | None:
    """Build a semantic search index over all tables in the warehouse."""
    import os

    if not os.environ.get("VOYAGE_API_KEY"):
        return None  # graceful fallback if no Voyage key

    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        index = TableIndex()
        tables = con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
        ).fetchall()

        for schema, table in tables:
            cols = con.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                ORDER BY ordinal_position
                """,
                [schema, table],
            ).fetchall()
            col_names = [c[0] for c in cols]

            try:
                row_count = con.execute(
                    f'SELECT COUNT(*) FROM "{schema}"."{table}"'
                ).fetchone()[0]
            except Exception:
                row_count = 0

            index.add_table(schema, table, col_names, row_count)

        index.build()
        return index
    except Exception:
        return None  # graceful fallback
    finally:
        con.close()
