"""Generate manuscript figures from the real pipeline outputs."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

BASE = cfg.PROJECT_ROOT
FIG = cfg.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)
RL = cfg.RL_DIR
REPORTS = cfg.REPORTS_DIR
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

BLUE, ORANGE, GREY, RED = "#2b6cb0", "#dd6b20", "#718096", "#c53030"


# ---------- Fig 1: glucose distribution with clinical zones ----------
def fig_glucose():
    ep = pd.read_parquet(BASE / "data/processed/episodes.parquet")
    glc = ep["last_glc"].dropna()
    glc = glc[(glc > 0) & (glc < 500)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(glc, bins=80, color=BLUE, alpha=0.85)
    ax.axvspan(70, 180, color="green", alpha=0.08, label="target 70-180")
    ax.axvline(70, color=RED, lw=1, ls="--")
    ax.axvline(180, color=ORANGE, lw=1, ls="--")
    ax.set_xlabel("blood glucose (mg/dL)")
    ax.set_ylabel("count (state-bins)")
    ax.set_title("Glucose readings across the ICU cohort")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig1_glucose.png"); plt.close(fig)


# ---------- Fig 2: cohort composition (care unit + age) ----------
def fig_cohort():
    ep = pd.read_parquet(BASE / "data/processed/episodes_enriched.parquet")
    stay = ep.drop_duplicates("ICUSTAY_ID")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    cu = stay["care_unit"].value_counts()
    axes[0].bar(cu.index, cu.values, color=BLUE)
    axes[0].set_title("ICU stays by first care unit")
    axes[0].set_ylabel("stays")
    for lbl in axes[0].get_xticklabels():
        lbl.set_rotation(30)
    axes[1].hist(stay["age"], bins=30, color=GREY)
    axes[1].set_title("Age at ICU admission")
    axes[1].set_xlabel("years (capped at 90)")
    axes[1].set_ylabel("stays")
    fig.tight_layout(); fig.savefig(FIG / "fig2_cohort.png"); plt.close(fig)


# ---------- Fig 3: MDP schematic ----------
def fig_mdp():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="black", lw=1.2, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    box(0.02, 0.35, 0.24, 0.34,
        "STATE (bin t)\nglucose, slope,\nbasal-24h context,\nage / sex / care unit", "#bee3f8")
    box(0.38, 0.35, 0.22, 0.34,
        "ACTION (bin t)\nnone / SC bolus /\nSC push / IV infusion\n× dose tertile", "#feebc8")
    box(0.72, 0.35, 0.24, 0.34,
        "REWARD\nnext-bin glucose\n(proxy) + terminal\nsurvival", "#c6f6d5")
    ax.annotate("", xy=(0.38, 0.52), xytext=(0.26, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.72, 0.52), xytext=(0.60, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("4-hour step -> bin t+1", xy=(0.5, 0.12), ha="center", fontsize=9,
                xytext=(0.5, 0.12))
    ax.annotate("", xy=(0.02, 0.28), xytext=(0.84, 0.28),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=GREY, connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.8)
    ax.set_title("Insulin-dosing MDP: basal as context, reactive correction as action")
    fig.tight_layout(); fig.savefig(FIG / "fig3_mdp.png"); plt.close(fig)


# ---------- Fig 4: clinician vs learned action distribution ----------
def fig_actions():
    meta = json.loads((RL / "meta.json").read_text())
    route_short = {"NONE": "none", "BOLUS_INYECTION": "SCbolus",
                   "BOLUS_PUSH": "IVbolus", "INFUSION": "IVinf"}
    labels = [f"{route_short[m['route']]}"
              f"{'' if m['dose_bin']<0 else '/'+['lo','md','hi'][m['dose_bin']]}"
              for m in meta["action_meta"]]
    test = np.load(RL / "test.npz")
    clin = np.bincount(test["action"], minlength=10) / len(test["action"])
    # learned greedy from enriched report action mix
    txt = (REPORTS / "cql_enriched.md").read_text()
    mix = [float(x) for x in re.findall(r"bin-?\d+\):\s+([\d.]+)%", txt)]
    mix = np.array(mix) / 100 if mix else np.zeros(10)
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - 0.2, clin, 0.4, label="clinician (logged)", color=GREY)
    ax.bar(x + 0.2, mix, 0.4, label="learned policy (greedy)", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("fraction of decisions")
    ax.set_title("Action distribution: clinician vs learned policy")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_actions.png"); plt.close(fig)


# ---------- Fig 5: training convergence ----------
def fig_training():
    txt = (REPORTS / "cql_enriched.md").read_text()
    epochs = [int(e) for e in re.findall(r"epoch\s+(\d+)\s+loss", txt)]
    qg = [float(q) for q in re.findall(r"val_Q\(greedy\)=([\d.]+)", txt)]
    td = [float(t) for t in re.findall(r"td=([\d.]+)", txt)]
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.plot(epochs, qg, "-o", color=BLUE, ms=3, label="val Q (greedy)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("Q value", color=BLUE)
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(epochs, td, "-s", color=ORANGE, ms=3, label="TD loss")
    ax2.set_ylabel("TD loss", color=ORANGE)
    ax1.set_title("CQL training: value and TD loss converge")
    fig.tight_layout(); fig.savefig(FIG / "fig5_training.png"); plt.close(fig)


# ---------- Fig 6: OPE forest plots (proxy + mortality) ----------
def _parse_ope(path):
    d = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(clinician_Vpi_b|FQE|WIS|WDR)\s*\|\s*([+\-\d.]+)\s*\|\s*\[([+\-\d.]+),\s*([+\-\d.]+)\]", line)
        if m:
            d[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return d


def fig_ope():
    proxy = _parse_ope(REPORTS / "ope_enriched_proxy.md")
    mort = _parse_ope(REPORTS / "ope_enriched_mortality.md")
    order = ["clinician_Vpi_b", "FQE", "WDR", "WIS"]
    names = ["clinician", "FQE", "WDR", "WIS"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, data, title in [(axes[0], proxy, "Glucose proxy reward"),
                            (axes[1], mort, "In-hospital mortality (terminal)")]:
        clin = data["clinician_Vpi_b"][0]
        for i, k in enumerate(order):
            pt, lo, hi = data[k]
            col = GREY if k == "clinician_Vpi_b" else (BLUE if k in ("FQE", "WDR") else ORANGE)
            ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3)
        ax.axvline(clin, color=GREY, ls="--", lw=1)
        ax.set_yticks(range(4)); ax.set_yticklabels(names)
        ax.invert_yaxis(); ax.set_title(title, fontsize=10)
        ax.set_xlabel("estimated policy value")
    fig.suptitle("Off-policy value estimates (95% CI); dashed = clinician", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig6_ope.png"); plt.close(fig)


# ---------- Fig 7: reward-sensitivity ablation ----------
def fig_ablation():
    txt = (REPORTS / "reward_ablation.md").read_text()
    rows = re.findall(r"\|\s*(piecewise|tir_binary|asymmetric|smooth)\s*\|\s*([+\-\d.]+)\s*\|\s*([+\-\d.]+)\s*\|\s*\[([+\-\d.]+),\s*([+\-\d.]+)\]", txt)
    labels = [r[0] for r in rows]
    clin = [float(r[1]) for r in rows]
    wdr = [float(r[2]) for r in rows]
    lo = [float(r[3]) for r in rows]; hi = [float(r[4]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - 0.2, clin, 0.4, label="clinician", color=GREY)
    ax.errorbar(x + 0.2, wdr, yerr=[np.array(wdr) - lo, np.array(hi) - np.array(wdr)],
                fmt="o", color=BLUE, capsize=3, label="learned (WDR)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("policy value")
    ax.set_title("Reward-sensitivity: learned vs clinician across 4 rewards")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig7_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    fig_glucose(); fig_cohort(); fig_mdp(); fig_actions()
    fig_training(); fig_ope(); fig_ablation()
    print("wrote figures to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print(" ", p.name)
