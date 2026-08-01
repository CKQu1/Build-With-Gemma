from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unzip the NASA SMAP/MSL dataset from Kaggle.")
    parser.add_argument("--output", default="data/raw/nasa", help="Directory where the dataset will be extracted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(output),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    print(f"Dataset extracted to {output.resolve()}")


if __name__ == "__main__":
    main()

