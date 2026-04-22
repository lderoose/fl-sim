#!/usr/bin/env python3
"""Entry point for fl-sim.

Run a single federated learning experiment:
    python main.py
    python main.py --path_config conf/application.py
    python main.py --path_config conf/application.py --path_report output.json
"""
import argparse

from conf.paths import ROOT
from src.run_single import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated learning experiment.")
    parser.add_argument(
        "--path_config",
        "-c",
        default=str(ROOT / "conf/application.py"),
        type=str,
        help="path to the configuration file (default: conf/application.py)",
    )
    parser.add_argument(
        "--path_report",
        type=str,
        default=None,
        help="optional path to override the report save location defined in the config",
    )
    args = parser.parse_args()
    main(path_config=args.path_config, path_report=args.path_report)
