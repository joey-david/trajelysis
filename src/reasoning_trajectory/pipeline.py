"""End-to-end Hugging Face generation and trajectory analysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from reasoning_trajectory.analysis.answers import update_answers
from reasoning_trajectory.analysis.hard_questions import write_hard_questions
from reasoning_trajectory.analysis.solution_objects import write_solution_objects
from reasoning_trajectory.datasets.loaders import load_normalized_dataset
from reasoning_trajectory.generation import generate_one_run
from reasoning_trajectory.runtime.config import load_config
from reasoning_trajectory.runtime.data import write_jsonl


def prepare_run(config_path: str | Path, run_path: str | Path) -> Path:
    """Create a self-contained run folder and materialize its selected dataset."""
    config_path = Path(config_path).expanduser().resolve()
    run_path = Path(run_path).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    destination = run_path / "config.yaml"
    if config_path != destination:
        shutil.copy2(config_path, destination)

    config = load_config(run_path)
    samples = load_normalized_dataset(config["dataset"])
    write_jsonl(run_path / "dataset.jsonl", samples)
    print(f"prepared {len(samples)} samples in {run_path}")
    return run_path


def analyze_run(run_path: str | Path) -> Path:
    """Derive analysis artifacts from a completed generation run."""
    run_path = Path(run_path).expanduser().resolve()
    config = load_config(run_path)
    analysis_cfg = config.get("analysis", {})

    update_answers(run_path, analysis_cfg)
    write_solution_objects(run_path, analysis_cfg)
    write_hard_questions(run_path, analysis_cfg)

    # Richer analyzers are optional while the public package is being split
    # out of the research repository. When present, run them automatically.
    try:
        from reasoning_trajectory.analysis.step_markers import write_step_markers

        write_step_markers(run_path, analysis_cfg)
    except ImportError:
        pass

    try:
        from reasoning_trajectory.analysis.interactive_trajectories import (
            write_interactive_trajectories,
        )

        write_interactive_trajectories(run_path, analysis_cfg)
    except ImportError:
        pass

    try:
        from reasoning_trajectory.analysis.step_classification import (
            write_step_classification,
        )

        write_step_classification(run_path, analysis_cfg)
    except ImportError:
        pass

    print(f"analyzed {run_path}")
    return run_path


def run(config_path: str | Path, run_path: str | Path) -> Path:
    """Prepare data, generate repeated rollouts, and analyze the resulting run."""
    run_path = prepare_run(config_path, run_path)
    generate_one_run(run_path)
    return analyze_run(run_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajelysis",
        description="Generate and analyze layer-wise latent reasoning trajectories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="prepare, generate, and analyze a run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--out", required=True)

    prepare_parser = sub.add_parser("prepare", help="materialize the configured dataset")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--out", required=True)

    generate_parser = sub.add_parser("generate", help="generate or resume a prepared run")
    generate_parser.add_argument("run_path")

    analyze_parser = sub.add_parser("analyze", help="analyze an existing generated run")
    analyze_parser.add_argument("run_path")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        run(args.config, args.out)
    elif args.command == "prepare":
        prepare_run(args.config, args.out)
    elif args.command == "generate":
        generate_one_run(Path(args.run_path))
    elif args.command == "analyze":
        analyze_run(args.run_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
