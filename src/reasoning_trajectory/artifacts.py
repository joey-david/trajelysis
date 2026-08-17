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
    """Read sample metadata, falling back to the pinned dataset for older runs."""
    run_path = Path(run_path)
    records: dict[str, dict[str, Any]] = {}

    dataset_path = run_path / "dataset.jsonl"
    if dataset_path.exists():
        for row in load_samples(dataset_path.resolve()):
            sample_id = row.get("sample_id") or row.get("id") or row.get("problem_id")
            if sample_id is not None:
                records[str(sample_id)] = {**row, "sample_id": str(sample_id)}

    sample_dir = run_path / "generation" / "samples"
    for path in sample_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(record.get("sample_id", path.stem))
        # Generation-time records are authoritative, but dataset fields such as
        # gold_answer and question remain available when old schemas omitted them.
        records[sample_id] = {**records.get(sample_id, {}), **record, "sample_id": sample_id}

    return records


__all__ = ["load_hidden_states_npz", "read_generation_rows", "read_sample_records"]
