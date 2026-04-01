"""Claude tool-use agent loop with streaming output."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from agent.context import build_system_prompt
from agent.tools import (
    TOOL_DEFINITIONS,
    describe_table,
    execute_sql,
    get_metadata_context,
    list_tables,
)

MODEL = "claude-sonnet-4-6"


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

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic()
        self.system_prompt = build_system_prompt(self.db_path, self.project_dir)

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
        self.messages.append({"role": "user", "content": user_message})

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
