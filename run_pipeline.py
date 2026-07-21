"""
Run the whole study end to end.

    python run_pipeline.py              # full pipeline
    python run_pipeline.py --quick      # small subset, for a smoke test
    python run_pipeline.py --from 5     # resume from a given stage
    python run_pipeline.py --only 6     # run a single stage

Stages map one-to-one onto the paper:

    1  build episodes      raw events  -> 4-hour decision bins        (Sec. 4.1-4.2)
    2  add outcomes        link MIMIC-III mortality and demographics  (Sec. 3)
    3  prepare dataset     discretize actions, standardize state      (Sec. 4.2)
    4  behavior cloning    sanity check on the state representation   (Sec. 4.4, 5.1)
    5  train CQL           conservative offline RL policy             (Sec. 4.3, 5.2)
    6  evaluate OPE        four estimators vs clinicians              (Sec. 4.5, 5.3)
    7  reward ablation     retrain and re-evaluate under four rewards (Sec. 4.6, 5.4)
    8  make figures        figures used in the paper                  (Figures 1-7)
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
PY = sys.executable

STAGES = [
    (1, "build episodes", "s1_build_episodes.py", []),
    (2, "add outcomes", "s2_add_outcomes.py", []),
    (3, "prepare dataset", "s3_prepare_dataset.py", []),
    (4, "behavior cloning", "s4_behavior_cloning.py", []),
    (5, "train CQL", "s5_train_cql.py", []),
    (6, "evaluate OPE", "s6_evaluate_ope.py", []),
    (7, "reward ablation", "s7_reward_ablation.py", []),
    (8, "make figures", "s8_make_figures.py", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first stage to run")
    ap.add_argument("--to", dest="end", type=int, default=8, help="last stage to run")
    ap.add_argument("--only", type=int, default=None, help="run just this stage")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: build only 200 ICU stays in stage 1")
    args = ap.parse_args()

    start, end = (args.only, args.only) if args.only else (args.start, args.end)

    for num, name, script, extra in STAGES:
        if not (start <= num <= end):
            continue
        cmd = [PY, str(SRC / script)] + extra
        if args.quick and num == 1:
            cmd += ["--limit-icustays", "200"]
        print(f"\n{'=' * 62}\n  stage {num}  |  {name}\n{'=' * 62}", flush=True)
        subprocess.run(cmd, check=True)

    print("\npipeline finished")


if __name__ == "__main__":
    main()
