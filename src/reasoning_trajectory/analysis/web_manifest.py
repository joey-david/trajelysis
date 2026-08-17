"""Discover analyzed runs and write the manifest consumed by the static web interface."""

from __future__ import annotations

import json
from pathlib import Path

from reasoning_trajectory.analysis.common import read_sample_records
from reasoning_trajectory.runtime.config import load_config


def write_manifest(runs_root: Path, out_path: Path) -> None:
    runs = []
    for config in sorted(runs_root.glob("*/**/config.yaml")):
        run = config.parent
        relative = run.relative_to(runs_root)
        generation_artifact = discover_generation_artifact(run)
        if generation_artifact is None:
            continue
        gen, generation_format, samples = generation_artifact
        plots = load_json(run / "analysis" / "plots" / "index.json", [])
        interactive_plots = load_json(
            run / "analysis" / "plots" / "interactive_index.json", []
        )
        step_classification_plots = load_json(
            run / "analysis" / "step_classification" / "interactive_index.json", []
        )
        step_markers = run / "analysis" / "step_markers.json"
        solution_objects = run / "analysis" / "solution_objects.jsonl"
        hard_questions = run / "analysis" / "hard_questions.jsonl"
        runs.append(
            {
                "model": relative.parts[0],
                "run": "/".join(relative.parts[1:]),
                "generations": web_path(gen),
                "generation_format": generation_format,
                "samples": samples,
                "plots": add_web_paths(run, plots),
                "interactive_plots": add_web_paths(run, interactive_plots),
                "step_classification_plots": add_web_paths(run, step_classification_plots),
                "step_markers": web_path(step_markers) if step_markers.exists() else None,
                "solution_objects": web_path(solution_objects) if solution_objects.exists() else None,
                "hard_questions": web_path(hard_questions) if hard_questions.exists() else None,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs}, ensure_ascii=False), encoding="utf-8")


def discover_generation_artifact(
    run: Path,
) -> tuple[Path, str, dict[str, dict]] | None:
    generations = run / "generation" / "generations.jsonl"
    if generations.exists():
        return generations, "generation", read_sample_records(run)

    continuations = run / "patching" / "continuations.jsonl"
    if not continuations.exists():
        return None
    activation_run = Path(load_config(run)["patching"]["activation_run"])
    return continuations, "causal_patching", read_sample_records(activation_run)


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def web_path(path: Path) -> str:
    return "../" + path.as_posix()


def add_web_paths(run: Path, plots: list[dict]) -> list[dict]:
    return [{**plot, "path": web_path(run / plot["path"])} for plot in plots]
