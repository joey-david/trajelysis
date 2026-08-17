"""Approximate generated-token character spans without loading a tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


TokenSpan = tuple[int, int] | None


def build_token_spans(
    run_path: Path,
    rows: list[dict[str, Any]],
) -> list[list[TokenSpan]]:
    """Map token indices to approximate character spans using stored text only.

    Analysis of an existing run is self-contained: opening the web UI must not
    contact Hugging Face or reload a model/tokenizer. Exact token boundaries are
    therefore traded for deterministic proportional spans.
    """
    del run_path
    return [approximate_token_spans(row) for row in rows]


def approximate_token_spans(row: dict[str, Any]) -> list[TokenSpan]:
    """Return deterministic proportional character spans for stored tokens."""
    token_count = len(row.get("generated_token_ids", []))
    text = str(row.get("produced_text", ""))
    if token_count <= 0:
        return []
    if not text:
        return [None] * token_count

    text_len = len(text)
    spans: list[TokenSpan] = []
    for token_idx in range(token_count):
        start = round(token_idx * text_len / token_count)
        end = round((token_idx + 1) * text_len / token_count)
        spans.append((start, max(start, end)))
    return spans


def token_range_for_chars(
    spans: list[TokenSpan],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    indices = [
        token_idx
        for token_idx, span in enumerate(spans)
        if span is not None and span[1] > char_start and span[0] < char_end
    ]
    return (indices[0], indices[-1]) if indices else None
