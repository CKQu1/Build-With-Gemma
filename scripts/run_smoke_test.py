from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "deep_space_navigation.train", "--config", "configs/smoke.yaml"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_space_navigation.infer",
            "--checkpoint",
            "outputs/smoke/best_model.pt",
            "--make-plots",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

