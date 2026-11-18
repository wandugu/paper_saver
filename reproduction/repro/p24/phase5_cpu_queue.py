"""Run explicit CPU-only phase5 experiment lists through repro.run."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("pattern")
    parser.add_argument("seeds", nargs="+", type=int)
    args = parser.parse_args()
    runs = Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase5"
    for seed in args.seeds:
        command = [sys.executable, "-u", "-m", "repro.run", "P24", args.experiment, "--config", args.pattern.format(seed=seed),
                   "--seed", str(seed), "--runs-root", str(runs), "--prereg", "repro/p24/prereg_p24_phase5.yaml"]
        subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)


if __name__ == "__main__":
    main()
