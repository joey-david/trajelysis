"""Shared readers for generated run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reasoning_trajectory.runtime.artifact_store import load_hidden_states_npz
from reasoning_trajectory.runtime.data import load_samples


def read_generation_rows(run_path: str | Path) -> list[dict[str, Any]]:
    """Read all generation rows from a completed run."""
    path = Path(run_path) / "generation" / "generations.jsonl"
    return load_samples(path.resolve())


def read_sample_records(run_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read persisted sample metadata keyed by the original sample ID."""
    sample_dir = Path(run_path) / "generation" / "samples"
    records: dict[str, dict[str, Any]] = {}
    for path in sample_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(record.get("sample_id", path.stem))
        records[sample_id] = record
    return records


__all__ = ["load_hidden_states_npz", "read_generation_rows", "read_sample_records"]
