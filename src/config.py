"""
Central configuration: every path and hyperparameter the pipeline uses.

Generated artifacts (episodes, tensors, models, reports, figures) live inside the
project folder. The two inputs that may not be redistributed are read from
locations you can set with environment variables:

    GLYCEMIC_RAW_DIR   folder holding glucose_insulin_pair.csv
    MIMIC_DIR          folder holding ADMISSIONS.csv, PATIENTS.csv, ICUSTAYS.csv

PowerShell example:
    $env:GLYCEMIC_RAW_DIR = "C:\\data\\glucose-management-mimic"
    $env:MIMIC_DIR        = "C:\\data\\mimic-iii"

Bash example:
    export GLYCEMIC_RAW_DIR=/data/glucose-management-mimic
    export MIMIC_DIR=/data/mimic-iii

If unset, they default to data/raw and data/mimic inside the project.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- inputs ----
# Credentialed data. Never stored in this repository (see README, Data access).
RAW_DIR = Path(os.environ.get("GLYCEMIC_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", PROJECT_ROOT / "data" / "mimic"))

PAIR_CSV = RAW_DIR / "glucose_insulin_pair.csv"
ADMISSIONS_CSV = MIMIC_DIR / "ADMISSIONS.csv"
PATIENTS_CSV = MIMIC_DIR / "PATIENTS.csv"
ICUSTAYS_CSV = MIMIC_DIR / "ICUSTAYS.csv"

# ---------------------------------------------------- generated artifacts ----
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RL_DIR = DATA_DIR / "rl"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

EPISODES = PROCESSED_DIR / "episodes.parquet"
EPISODES_ENRICHED = PROCESSED_DIR / "episodes_enriched.parquet"

# --------------------------------------------- decision process (Sec. 4.1) ----
BIN_HOURS = 4          # length of one decision step, in hours
GAMMA = 0.95           # discount per 4-hour step
SEED = 0
TRAIN_FRAC = 0.70      # patient-level split
VAL_FRAC = 0.15

# ------------------------------------------- policy learning, CQL (Sec. 4.3) ----
CQL_ALPHA = 1.0        # weight on the conservative penalty
CQL_EPOCHS = 60
CQL_LR = 1e-3
CQL_BATCH = 512
CQL_TARGET_TAU = 0.005  # Polyak averaging rate for the target network
CQL_HIDDEN = 128

# ------------------------------------ off-policy evaluation (Sec. 4.5) ----
OPE_TAU = 0.5           # softmax temperature for the target policy
OPE_RATIO_CLIP = 5.0    # clip on the per-step log importance ratio
OPE_PROB_FLOOR = 1e-3   # floor on estimated behavior-policy probability
OPE_N_BOOTSTRAP = 200   # patient-level bootstrap resamples
FQE_EPOCHS = 40


def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for d in (PROCESSED_DIR, RL_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
