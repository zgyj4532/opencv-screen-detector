"""Run the predeclared six training/validation jobs serially, with durable logs."""

import argparse
import subprocess
import sys

from experiment.daily_20260922.src.protocol import (
    LOGS,
    REPORTS,
    ROOT,
    SEEDS,
    VARIANTS,
    candidate_name,
    prepare_directories,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume unfinished training; retain completed evidence")
    args = parser.parse_args()
    prepare_directories()
    for seed in SEEDS:
        for variant in VARIANTS:
            name = candidate_name(variant, seed)
            for stage in ("train", "val"):
                if (REPORTS / f"{name}.{stage}.json").exists():
                    print(f"Existing {name} {stage}; retained", flush=True)
                    continue
                module = "train" if stage == "train" else "evaluate"
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    f"experiment.daily_20260922.src.{module}",
                    variant,
                    "--seed",
                    str(seed),
                ]
                if (
                    stage == "train"
                    and args.resume
                    and (ROOT / f"experiment/cnn_fft_dwt_ablation/exp/{name}/last.pth").exists()
                ):
                    command.append("--resume")
                print(f"Starting {name} {stage}", flush=True)
                with (LOGS / f"{name}.{stage}.log").open("a", encoding="utf-8") as handle:
                    subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)  # noqa: S603
                print(f"Completed {name} {stage}", flush=True)


if __name__ == "__main__":
    main()
