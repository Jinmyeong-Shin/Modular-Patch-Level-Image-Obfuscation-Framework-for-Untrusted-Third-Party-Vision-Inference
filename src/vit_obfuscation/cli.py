from __future__ import annotations

import argparse
import logging
import sys

from .config.experiment import ExperimentConfig
from .runner.runner import ExperimentRunner, run_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description="ViT Obfuscation Framework - Experiment Runner",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run single experiment
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment YAML config",
    )
    run_parser.add_argument("--output-dir", "-o", help="Override output directory")
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    # Run all experiments
    all_parser = subparsers.add_parser(
        "run-all", help="Run all experiments in optimal order"
    )
    all_parser.add_argument(
        "--config-dir",
        "-d",
        default="configs/experiments",
        help="Directory containing experiment YAML configs",
    )
    all_parser.add_argument(
        "--output-dir", "-o", default="./outputs", help="Output directory"
    )
    all_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy HTTP logs unless truly verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        if args.output_dir:
            config.output_dir = args.output_dir

        runner = ExperimentRunner(config)
        results = runner.run()

        print("\n=== Results ===")
        print(f"Experiment: {results['experiment']}")
        print(f"Model: {results['model']}")
        print(f"Task: {results['task']}")
        print(f"\nClean:      {results['clean']}")
        print(f"Obfuscated: {results['obfuscated']}")

    elif args.command == "run-all":
        all_results = run_all_experiments(args.config_dir, args.output_dir)

        print("\n=== All Results ===")
        for r in all_results:
            if "error" in r:
                print(f"\n[FAILED] {r['experiment']}: {r['error']}")
            else:
                print(f"\n[OK] {r['experiment']} ({r['model']} / {r['task']})")
                print(f"  Clean:      {r['clean']}")
                print(f"  Obfuscated: {r['obfuscated']}")


if __name__ == "__main__":
    main()
