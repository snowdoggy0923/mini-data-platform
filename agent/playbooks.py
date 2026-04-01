"""Playbooks — cache and retrieve successful query patterns.

Inspired by Kepler's playbook system: when the agent successfully answers
a question, save the pattern (question + SQL + tables used). On future
similar questions, surface the relevant playbook to help the agent respond
faster and more accurately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from agent.search import _voyage_embed
SIMILARITY_THRESHOLD = 0.75  # minimum similarity to consider a playbook relevant


class PlaybookStore:
    """Stores and retrieves successful query patterns."""

    def __init__(self, store_dir: str) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._playbooks: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, question: str, sql_queries: list[str], tables_used: list[str]) -> None:
        """Save a successful query pattern as a playbook."""
        embedding = self._embed(question)

        playbook = {
            "question": question,
            "sql_queries": sql_queries,
            "tables_used": tables_used,
            "embedding": embedding.tolist(),
            "created_at": time.time(),
        }

        self._playbooks.append(playbook)
        self._rebuild_index()
        self._persist(playbook)

    def find_similar(self, question: str, top_k: int = 3) -> list[dict]:
        """Find playbooks similar to the given question.

        Returns list of playbooks (without embeddings) sorted by relevance,
        only if similarity exceeds SIMILARITY_THRESHOLD.
        """
        if not self._playbooks or self._embeddings is None:
            return []

        query_emb = self._embed(question)
        similarities = self._embeddings @ query_emb
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        similarities = similarities / np.where(norms == 0, 1, norms)

        ranked = np.argsort(similarities)[::-1][:top_k]
        results = []
        for i in ranked:
            score = float(similarities[i])
            if score < SIMILARITY_THRESHOLD:
                break
            pb = {k: v for k, v in self._playbooks[i].items() if k != "embedding"}
            pb["similarity"] = score
            results.append(pb)

        return results

    def format_for_prompt(self, playbooks: list[dict]) -> str:
        """Format matched playbooks into text for injection into the conversation."""
        if not playbooks:
            return ""

        parts = ["[Relevant playbooks from previous successful queries]:"]
        for i, pb in enumerate(playbooks, 1):
            sql_str = "\n    ".join(pb["sql_queries"])
            parts.append(
                f"\n{i}. Similar question: \"{pb['question']}\" "
                f"(similarity: {pb['similarity']:.0%})\n"
                f"   Tables used: {', '.join(pb['tables_used'])}\n"
                f"   SQL patterns:\n    {sql_str}"
            )

        parts.append(
            "\nUse these as reference patterns — adapt the SQL to the current question, "
            "don't copy blindly."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        return np.array(_voyage_embed([text], input_type="query")[0])

    def _rebuild_index(self) -> None:
        if not self._playbooks:
            self._embeddings = None
            return
        self._embeddings = np.array([pb["embedding"] for pb in self._playbooks])

    def _persist(self, playbook: dict) -> None:
        """Append a playbook to the store file."""
        store_file = self._store_dir / "playbooks.jsonl"
        with open(store_file, "a") as f:
            f.write(json.dumps(playbook) + "\n")

    def _load(self) -> None:
        """Load existing playbooks from disk."""
        store_file = self._store_dir / "playbooks.jsonl"
        if not store_file.exists():
            return

        for line in store_file.read_text().strip().split("\n"):
            if line:
                try:
                    self._playbooks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self._rebuild_index()
