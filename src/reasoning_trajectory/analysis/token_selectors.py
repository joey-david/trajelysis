"""Build token-index selectors for regular intervals, text boundaries, and regex regions."""

from __future__ import annotations

import re
from typing import Any, Callable


Selector = Callable[[dict[str, Any]], list[int]]


def build_token_selector(spec: dict[str, Any] | None) -> Selector:
    spec = spec or {"every_n": 1}
    mode = spec.get("mode")

    if mode == "sentence_end" or spec.get("sentence_end"):
        return sentence_end_tokens
    if mode == "percentiles" or "percentiles" in spec:
        values = spec.get("percentiles")
        if values is None:
            count = max(int(spec.get("count", spec.get("P", 10))), 1)
            values = [i * 100.0 / count for i in range(count + 1)]
        return lambda row: percentile_tokens(row, values)
    if mode == "reasoning_boundaries" or spec.get("reasoning_boundaries"):
        return reasoning_boundary_tokens
    if mode == "first_last" or spec.get("first_last"):
        return first_last_tokens
    if "before_regex" in spec or "after_regex" in spec:
        pattern = spec.get("before_regex") or spec.get("after_regex")
        after = "after_regex" in spec
        return lambda row: regex_tokens(row, pattern, after)

    n = max(int(spec.get("every_n", 1)), 1)
    return lambda row: every_n_tokens(row, n)


def every_n_tokens(row: dict[str, Any], n: int) -> list[int]:
    return list(range(0, token_count(row), max(int(n), 1)))


def sentence_end_tokens(row: dict[str, Any]) -> list[int]:
    text = row.get("produced_text", "")
    if not text:
        return []
    return unique_existing_tokens(
        row, [char_to_token_index(row, pos) for pos in sentence_end_positions(text)]
    )


def sentence_end_positions(text: str) -> list[int]:
    positions: list[int] = []
    for match in re.finditer(r"\.+", text):
        start, end = match.span()
        is_decimal = (
            end - start == 1
            and start > 0
            and end < len(text)
            and text[start - 1].isdigit()
            and text[end].isdigit()
        )
        if not is_decimal:
            positions.append(end)
    return positions


def percentile_tokens(row: dict[str, Any], percentiles: list[int | float]) -> list[int]:
    total = token_count(row)
    if total <= 0:
        return []
    return unique_existing_tokens(
        row, [round((float(p) / 100.0) * (total - 1)) for p in percentiles]
    )


def reasoning_boundary_tokens(row: dict[str, Any]) -> list[int]:
    total = token_count(row)
    if total <= 0:
        return []
    reasoning_length = row.get("reasoning_length")
    if reasoning_length is None:
        return first_last_tokens(row)
    boundary = min(max(int(reasoning_length) - 1, 0), total - 1)
    return unique_existing_tokens(
        row, [0, boundary, min(boundary + 1, total - 1), total - 1]
    )


def first_last_tokens(row: dict[str, Any]) -> list[int]:
    total = token_count(row)
    if total <= 0:
        return []
    return unique_existing_tokens(row, [0, total - 1])


def regex_tokens(row: dict[str, Any], pattern: str, after: bool) -> list[int]:
    text = row.get("produced_text", "")
    match = re.search(pattern, text, re.S)
    if not match:
        return []
    pos = match.end() if after else match.start()
    token_idx = char_to_token_index(row, pos)
    total = token_count(row)
    return list(range(token_idx, total)) if after else list(range(token_idx))


def char_to_token_index(row: dict[str, Any], char_pos: int) -> int:
    text = row.get("produced_text", "")
    total = token_count(row)
    if total <= 0:
        return 0
    if not text:
        return min(max(char_pos, 0), total - 1)
    ratio = min(max(char_pos, 0), len(text)) / max(len(text), 1)
    return min(max(round(ratio * (total - 1)), 0), total - 1)


def token_count(row: dict[str, Any]) -> int:
    return len(row.get("generated_token_ids", []))


def unique_existing_tokens(row: dict[str, Any], indices: list[int]) -> list[int]:
    total = token_count(row)
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if 0 <= idx < total and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out
