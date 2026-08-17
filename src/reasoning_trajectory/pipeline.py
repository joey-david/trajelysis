"""End-to-end Hugging Face generation, analysis, and local exploration."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from reasoning_trajectory.analysis.answers import update_answers
from reasoning_trajectory.analysis.hard_questions import write_hard_questions
from reasoning_trajectory.analysis.interactive_trajectories import write_interactive_trajectories
from reasoning_trajectory.analysis.solution_objects import write_solution_objects
from reasoning_trajectory.analysis.step_classification import write_step_classification
from reasoning_trajectory.analysis.step_markers import write_step_markers
from reasoning_trajectory.artifacts import read_generation_rows, read_sample_records
from reasoning_trajectory.datasets.loaders import load_normalized_dataset
from reasoning_trajectory.generation import generate_one_run
from reasoning_trajectory.runtime.config import load_config
from reasoning_trajectory.runtime.data import write_jsonl


DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_PORT = 8765


def config_run_path(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} does not exist. Run ./setup or pass a config path."
        )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    name = str(data.get("name") or config_path.stem)
    return Path("runs") / name


def _generation_file(run_path: str | Path) -> Path:
    return Path(run_path) / "generation" / "generations.jsonl"


def _has_generation_rows(run_path: str | Path) -> bool:
    path = _generation_file(run_path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def _has_hidden_states(run_path: Path) -> bool:
    return any(
        row.get("hidden_states_file") and (run_path / row["hidden_states_file"]).exists()
        for row in read_generation_rows(run_path)
    )


def resolve_run_path(run_path: str | Path | None = None) -> Path:
    """Resolve an explicit run or automatically select a populated local run."""
    if run_path is not None:
        candidate = Path(run_path).expanduser().resolve()
        if not _has_generation_rows(candidate):
            raise FileNotFoundError(
                f"No generated traces found in {candidate}. "
                "Pass a run containing a non-empty generation/generations.jsonl."
            )
        return candidate

    if DEFAULT_CONFIG.exists():
        candidate = config_run_path(DEFAULT_CONFIG).resolve()
        if _has_generation_rows(candidate):
            return candidate

    candidates = []
    for generations in Path("runs").glob("**/generation/generations.jsonl"):
        run = generations.parent.parent
        if _has_generation_rows(run):
            candidates.append(generations)

    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        selected = newest.parent.parent.resolve()
        print(f"auto-selected populated run: {selected}")
        return selected

    raise FileNotFoundError(
        "No populated run found under runs/. `trajelysis web` needs at least one "
        "non-empty generation/generations.jsonl, or an explicit run directory."
    )


def prepare_run(
    config_path: str | Path = DEFAULT_CONFIG,
    run_path: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    if run_path is None:
        run_path = config_run_path(config_path)
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
    """Rebuild all configured analysis artifacts for a completed run."""
    run_path = Path(run_path).expanduser().resolve()
    config = load_config(run_path)
    analysis_cfg = config.get("analysis", {})

    update_answers(run_path, analysis_cfg)
    write_step_markers(run_path, analysis_cfg)
    write_solution_objects(run_path, analysis_cfg)
    write_hard_questions(run_path, analysis_cfg)

    if analysis_cfg.get("static_plots", False):
        from reasoning_trajectory.analysis.trajectories import plot_trajectories

        plot_trajectories(run_path, analysis_cfg)
    else:
        static_index = run_path / "analysis" / "plots" / "index.json"
        if static_index.exists():
            static_index.unlink()

    write_interactive_trajectories(run_path, analysis_cfg)
    write_step_classification(run_path, analysis_cfg)
    print(f"analyzed {run_path}")
    return run_path


def ensure_analysis(run_path: str | Path) -> Path:
    """Make an existing run useful in the web UI without needlessly rebuilding projections."""
    run_path = Path(run_path).expanduser().resolve()
    config = load_config(run_path)
    analysis_cfg = config.get("analysis", {})

    # These passes are cheap and keep correctness/summary metadata in sync.
    update_answers(run_path, analysis_cfg)
    write_step_markers(run_path, analysis_cfg)
    write_solution_objects(run_path, analysis_cfg)
    write_hard_questions(run_path, analysis_cfg)

    if _has_hidden_states(run_path):
        interactive_index = run_path / "analysis" / "plots" / "interactive_index.json"
        if not interactive_index.exists() or not _load_json(interactive_index, []):
            print("building missing latent projections...")
            write_interactive_trajectories(run_path, analysis_cfg)

        step_index = run_path / "analysis" / "step_classification" / "interactive_index.json"
        if not step_index.exists() or not _load_json(step_index, []):
            print("building missing step-classification projections...")
            write_step_classification(run_path, analysis_cfg)

    return run_path


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    run_path: str | Path | None = None,
) -> Path:
    run_path = prepare_run(config_path, run_path)
    generate_one_run(run_path)
    return analyze_run(run_path)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _web_paths(run_path: Path, entries: list[dict]) -> list[dict]:
    return [
        {**entry, "path": f"/run/{entry['path'].lstrip('/')}"}
        for entry in entries
    ]


def build_single_run_manifest(run_path: Path) -> dict:
    config = load_config(run_path)
    model_name = str(config.get("model", {}).get("name", "model")).split("/")[-1]
    samples = read_sample_records(run_path)

    plots = _load_json(run_path / "analysis" / "plots" / "index.json", [])
    interactive = _load_json(
        run_path / "analysis" / "plots" / "interactive_index.json", []
    )
    steps = _load_json(
        run_path / "analysis" / "step_classification" / "interactive_index.json", []
    )

    def artifact(relative: str) -> str | None:
        return f"/run/{relative}" if (run_path / relative).exists() else None

    return {
        "runs": [
            {
                "model": model_name,
                "run": run_path.name,
                "generations": "/run/generation/generations.jsonl",
                "generation_format": "generation",
                "samples": samples,
                "plots": _web_paths(run_path, plots),
                "interactive_plots": _web_paths(run_path, interactive),
                "step_classification_plots": _web_paths(run_path, steps),
                "step_markers": artifact("analysis/step_markers.json"),
                "solution_objects": artifact("analysis/solution_objects.jsonl"),
                "hard_questions": artifact("analysis/hard_questions.jsonl"),
                "trajectory_metrics": artifact("analysis/trajectory_metrics.json"),
            }
        ]
    }


def serve_web(
    run_path: str | Path | None = None,
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    analyze: bool = False,
) -> None:
    """Make a populated run viewable and serve it in the bundled interface."""
    run_path = resolve_run_path(run_path)
    if analyze:
        analyze_run(run_path)
    else:
        ensure_analysis(run_path)

    project_root = Path(__file__).resolve().parents[2]
    web_root = project_root / "web"
    if not (web_root / "index.html").exists():
        raise FileNotFoundError(
            f"Web interface not found at {web_root}. Run from an editable checkout."
        )

    manifest = json.dumps(build_single_run_manifest(run_path)).encode("utf-8")
    web_root = web_root.resolve()
    run_root = run_path.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/web/data/runs.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(manifest)))
                self.end_headers()
                self.wfile.write(manifest)
                return
            super().do_GET()

        def translate_path(self, url_path: str) -> str:
            path = unquote(urlsplit(url_path).path)
            if path == "/":
                return str(web_root / "index.html")
            if path.startswith("/web/"):
                return str(_safe_url_path(web_root, path[len("/web/") :]))
            if path.startswith("/run/"):
                return str(_safe_url_path(run_root, path[len("/run/") :]))
            return str(web_root / ".not-found")

        def guess_type(self, path: str) -> str:
            return mimetypes.guess_type(path)[0] or "application/octet-stream"

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/web/index.html"
    print(f"serving {run_path}")
    print(url)
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def _safe_url_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return root / ".not-found"
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajelysis",
        description="Generate and analyze layer-wise latent reasoning trajectories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="generate and analyze a run")
    run_parser.add_argument("config", nargs="?", default="config.yaml")
    run_parser.add_argument("out", nargs="?")

    web_parser = sub.add_parser("web", help="open a run in the web interface")
    web_parser.add_argument("run_path", nargs="?")
    web_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    web_parser.add_argument("--no-browser", action="store_true")
    web_parser.add_argument(
        "--analyze",
        action="store_true",
        help="force a complete rebuild of analysis artifacts before opening",
    )

    analyze_parser = sub.add_parser("analyze", help="rebuild analysis for an existing run")
    analyze_parser.add_argument("run_path", nargs="?")

    prepare_parser = sub.add_parser("prepare", help="materialize a config's dataset")
    prepare_parser.add_argument("config", nargs="?", default="config.yaml")
    prepare_parser.add_argument("out", nargs="?")

    generate_parser = sub.add_parser("generate", help="generate or resume a prepared run")
    generate_parser.add_argument("run_path", nargs="?")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        run(args.config, args.out)
    elif args.command == "web":
        serve_web(
            args.run_path,
            port=args.port,
            open_browser=not args.no_browser,
            analyze=args.analyze,
        )
    elif args.command == "analyze":
        analyze_run(resolve_run_path(args.run_path))
    elif args.command == "prepare":
        prepare_run(args.config, args.out)
    elif args.command == "generate":
        generate_one_run(resolve_run_path(args.run_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
