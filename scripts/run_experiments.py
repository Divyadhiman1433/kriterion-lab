from __future__ import annotations

import argparse
from pathlib import Path

from kriterion.experiments import run_all_experiments
from kriterion.visualization import generate_all_figures

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--figures-dir", default=str(ROOT / "assets" / "figures"))
    args = parser.parse_args()
    bundle = run_all_experiments(args.results_dir)
    generate_all_figures(bundle, args.figures_dir)
    print(f"Results: {Path(args.results_dir).resolve()}")
    print(f"Figures: {Path(args.figures_dir).resolve()}")


if __name__ == "__main__":
    main()
