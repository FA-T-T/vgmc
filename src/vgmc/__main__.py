"""Command-line access to the E1, E2 and E3 workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from .artifacts import read_json
from .runtime import RunStore
from .workflow import analyze, prepare_run, render, run_experiment


def main() -> None:
    """Parse one workflow stage and report its declared artifacts or status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "plan",
            "run",
            "status",
            "verify",
            "analyze",
            "render",
            "reproduce",
        ],
    )
    parser.add_argument("experiment", choices=["E1", "E2", "E3"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument(
        "--mode", choices=["fresh", "reuse", "resume", "auto"], default="fresh"
    )
    parser.add_argument("--allow-compute", action="store_true")
    parser.add_argument("--molecules", nargs="+")
    parser.add_argument("--covariances", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--without-random", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opts = dict(
        molecules=args.molecules,
        covariances=args.covariances,
        methods=args.methods,
        arms=args.arms,
        workers=args.workers,
        random_reference=not args.without_random,
        mode=args.mode,
        allow_compute=args.allow_compute,
    )
    if args.stage in ("status", "verify"):
        path = args.run or args.output
        frozen = read_json(path / "identity.json")
        if frozen["scientific"]["experiment"] != args.experiment:
            parser.error("Run experiment differs")
        store = RunStore(
            path, frozen["scientific"], frozen["tasks"], mode="auto"
        )
        decision = store.plan()
        if args.stage == "verify":
            store.verify()
        committed = sum(
            store.read_task(task) is not None for task in store.tasks
        )
        print(
            json.dumps(
                dict(
                    action=decision.action,
                    committed=committed,
                    expected=len(store.tasks),
                ),
                indent=2,
            )
        )
    elif args.stage == "plan":
        store, _ = prepare_run(args.experiment, args.root, args.output, **opts)
        print(
            json.dumps(
                dict(
                    decision=asdict(store.plan()),
                    binding=store.binding,
                    tasks=store.tasks,
                    effective=store.scientific,
                ),
                indent=2,
            )
        )
    elif args.stage == "run":
        frame = run_experiment(args.experiment, args.root, args.output, **opts)
        print(frame.to_string(index=False))
    elif args.stage in ("analyze", "reproduce"):
        analysis = args.analysis or args.output / "analysis"
        frame = analyze(args.experiment, args.root, analysis, run=args.run)
        print(frame.to_string(index=False))
        if args.stage == "reproduce":
            for path in render(args.experiment, analysis, args.output):
                print(path)
    else:
        if args.analysis is None:
            parser.error("Render requires --analysis")
        for path in render(args.experiment, args.analysis, args.output):
            print(path)


if __name__ == "__main__":
    main()
