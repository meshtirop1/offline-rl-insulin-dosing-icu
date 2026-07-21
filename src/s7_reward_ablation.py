"""
Reward-sensitivity ablation.

The single most common reason AI-Clinician-style ICU RL results get rejected or
retracted is that the headline conclusion (learned policy > clinician) flips when
the reward function is perturbed. This script tests exactly that: it re-derives
the reward from the stored next-bin glucose under several reward definitions,
retrains CQL from scratch under each, re-runs OPE, and reports whether the
DIRECTION of the policy-vs-clinician comparison is stable across all of them.

Reward variants (all functions of next-bin glucose):
  - piecewise   : the training default (build_episodes.reward_from_glc)
  - tir_binary  : +1 if 70..180, else 0  (pure time-in-range, no hyper/hypo grading)
  - asymmetric  : same as piecewise but with a much harsher hypoglycemia penalty
                  (severe hypo -2.0, hypo -1.0) -- tests robustness to how strongly
                  we punish the dangerous tail
  - smooth      : continuous Gaussian-shaped reward peaked at 110 mg/dL

We report a table: for each reward variant, clinician value, WDR value + CI, and
whether WDR's CI is above / overlaps / below the clinician. A trustworthy result
is one where the direction is the same across all variants.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as cfg

RL_DIR = cfg.RL_DIR
MODELS = cfg.MODELS_DIR
REPORTS = cfg.REPORTS_DIR
SRC = Path(__file__).parent
PY = sys.executable


def reward_piecewise(glc):
    r = np.where(np.isnan(glc), 0.0,
        np.select(
            [glc < 40, glc < 70, glc <= 180, glc <= 300],
            [-1.0, -0.4, 1.0, -0.3],
            default=-0.6,
        ))
    return r.astype(np.float32)


def reward_tir_binary(glc):
    r = np.where(np.isnan(glc), 0.0, np.where((glc >= 70) & (glc <= 180), 1.0, 0.0))
    return r.astype(np.float32)


def reward_asymmetric(glc):
    r = np.where(np.isnan(glc), 0.0,
        np.select(
            [glc < 40, glc < 70, glc <= 180, glc <= 300],
            [-2.0, -1.0, 1.0, -0.3],
            default=-0.6,
        ))
    return r.astype(np.float32)


def reward_smooth(glc):
    # continuous reward, peak +1 at 110, ~0 by 40/220, negative beyond
    r = np.where(np.isnan(glc), 0.0, 2.0 * np.exp(-((glc - 110.0) ** 2) / (2 * 45.0 ** 2)) - 1.0)
    return r.astype(np.float32)


VARIANTS = {
    "piecewise": reward_piecewise,
    "tir_binary": reward_tir_binary,
    "asymmetric": reward_asymmetric,
    "smooth": reward_smooth,
}


def rewrite_rewards(variant_fn):
    """Overwrite reward arrays in the train/val/test npz using next_glc."""
    backups = {}
    for split in ["train", "val", "test"]:
        p = RL_DIR / f"{split}.npz"
        d = dict(np.load(p))
        backups[split] = d["reward"].copy()
        d["reward"] = variant_fn(d["next_glc"])
        np.savez(p, **d)
    return backups


def restore_rewards(backups):
    for split, reward in backups.items():
        p = RL_DIR / f"{split}.npz"
        d = dict(np.load(p))
        d["reward"] = reward
        np.savez(p, **d)


def parse_ope_report(path):
    """Pull clinician value and WDR value+CI from an ope.md report."""
    text = path.read_text(encoding="utf-8")
    clin = wdr = wdr_lo = wdr_hi = None
    for line in text.splitlines():
        if line.startswith("| clinician_Vpi_b |"):
            parts = [p.strip() for p in line.split("|")]
            clin = float(parts[2])
        if line.startswith("| WDR |"):
            parts = [p.strip() for p in line.split("|")]
            wdr = float(parts[2])
            ci = parts[3].strip("[]").split(",")
            wdr_lo, wdr_hi = float(ci[0]), float(ci[1])
    return clin, wdr, wdr_lo, wdr_hi


def main():
    summary_rows = []
    original = None
    try:
        for name, fn in VARIANTS.items():
            print(f"\n===== reward variant: {name} =====")
            b = rewrite_rewards(fn)
            if original is None:
                original = b  # first backup is the true original piecewise reward

            model_path = MODELS / f"cql_{name}.pt"
            train_report = REPORTS / f"cql_{name}.md"
            ope_report = REPORTS / f"ope_{name}.md"

            subprocess.run([PY, str(SRC / "s5_train_cql.py"),
                            "--epochs", "40", "--out", str(model_path),
                            "--report", str(train_report)], check=True)
            subprocess.run([PY, str(SRC / "s6_evaluate_ope.py"),
                            "--qnet", str(model_path), "--report", str(ope_report)], check=True)

            clin, wdr, lo, hi = parse_ope_report(ope_report)
            direction = "ABOVE" if lo > clin else ("OVERLAPS" if hi > clin else "BELOW")
            summary_rows.append((name, clin, wdr, lo, hi, direction))
    finally:
        if original is not None:
            restore_rewards(original)  # leave npz with the default piecewise reward

    lines = ["# Reward-sensitivity ablation\n\n",
             "Each row retrains CQL from scratch under a different reward, then re-runs OPE.\n",
             "A trustworthy conclusion has the SAME direction across all variants.\n\n",
             "| reward variant | clinician V | WDR | WDR 95% CI | WDR vs clinician |\n",
             "|---|---|---|---|---|\n"]
    for name, clin, wdr, lo, hi, direction in summary_rows:
        lines.append(f"| {name} | {clin:+.4f} | {wdr:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {direction} |\n")

    directions = {r[5] for r in summary_rows}
    verdict = ("STABLE -- same direction across all reward variants"
               if len(directions) == 1 else
               "UNSTABLE -- direction flips across reward variants; do NOT report a benefit claim")
    lines.append(f"\n**Verdict: {verdict}**\n")

    out = REPORTS / "reward_ablation.md"
    out.write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
