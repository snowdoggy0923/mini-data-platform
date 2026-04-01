"""Semantic search for table discovery using VoyageAI embeddings.

Implements the "semantic embeddings" signal from Kepler's hybrid search.
At startup, embeds all table metadata. At query time, finds the most
relevant tables by cosine similarity.

Uses the VoyageAI REST API directly (not the SDK) for Python 3.14 compatibility.
"""
from __future__ import annotations

import os
import time

import httpx
import numpy as np

EMBED_MODEL = "voyage-3.5-lite"
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


def _voyage_embed(texts: list[str], input_type: str, max_retries: int = 3) -> list[list[float]]:
    """Call VoyageAI embedding API directly via HTTP, with retry on rate limit."""
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    for attempt in range(max_retries):
        response = httpx.post(
            VOYAGE_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts, "input_type": input_type},
            timeout=30.0,
        )
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
            continue
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    return []


class TableIndex:
    """A semantic search index over table metadata."""

    def __init__(self) -> None:
        self._documents: list[str] = []
        self._table_keys: list[str] = []
        self._embeddings: np.ndarray | None = None

    def add_table(self, schema: str, table: str, columns: list[str], row_count: int) -> None:
        """Add a table's metadata to the index (call before build())."""
        col_str = ", ".join(columns)
        doc = f"Table {schema}.{table} ({row_count:,} rows). Columns: {col_str}"
        self._documents.append(doc)
        self._table_keys.append(f"{schema}.{table}")

    def build(self) -> None:
        """Compute embeddings for all added tables."""
        if not self._documents:
            return
        embeddings = _voyage_embed(self._documents, input_type="document")
        self._embeddings = np.array(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find the most relevant tables for a natural language query.

        Returns list of (schema.table, similarity_score) sorted by relevance.
        """
        if self._embeddings is None or len(self._documents) == 0:
            return []

        query_embedding = np.array(_voyage_embed([query], input_type="query")[0])

        # Cosine similarity
        similarities = self._embeddings @ query_embedding
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = similarities / np.where(norms == 0, 1, norms)

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self._table_keys[i], float(similarities[i])) for i in ranked_indices]
