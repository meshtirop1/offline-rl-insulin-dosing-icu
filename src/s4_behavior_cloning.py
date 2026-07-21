"""
Behavior-cloning baseline: supervised models that predict the clinician's
actual action from state, factored the same way as the eventual RL policy
(route -> insulin type -> dose). This is a standard pre-RL sanity check:
  - if these models can't beat the majority-class baseline, the state
    representation in build_episodes.py is too weak for any offline-RL
    policy built on top of it to be trustworthy.
  - the trained BC policy also serves as a candidate "behavior policy"
    estimator for later off-policy evaluation (FQE / importance sampling
    both need one).

Uses HistGradientBoosting{Classifier,Regressor} for native NaN handling --
several state features (hours_since_last_glc, hours_since_last_dose,
glc_slope) are legitimately missing early in a stay, and imputing them
would inject an assumption we don't need to make yet.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight

STATE_COLS = [
    "last_glc", "glc_slope", "n_glc_readings_in_bin", "hours_since_last_glc",
    "hours_since_last_dose", "dose_last_24h", "los_icu_days", "first_icu_stay",
    "bin_index",
]

import config as cfg

DATA = cfg.EPISODES_ENRICHED if cfg.EPISODES_ENRICHED.exists() else cfg.EPISODES
OUT_DIR = cfg.MODELS_DIR
REPORT = cfg.REPORTS_DIR / "behavior_cloning.md"


def majority_baseline_report(y_true, majority_label):
    preds = np.full(len(y_true), majority_label)
    return classification_report(y_true, preds, zero_division=0)


def main():
    df = pd.read_parquet(DATA)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Behavior-cloning baseline\n"]
    lines.append(f"train={len(train)} val={len(val)} test={len(test)} rows\n")

    # ---- 1. route classifier (NONE / BOLUS_INYECTION / BOLUS_PUSH / INFUSION) ----
    X_train, y_train_route = train[STATE_COLS], train["route"]
    X_val, y_val_route = val[STATE_COLS], val["route"]
    X_test, y_test_route = test[STATE_COLS], test["route"]

    sw = compute_sample_weight("balanced", y_train_route)
    route_clf = HistGradientBoostingClassifier(random_state=0, max_iter=300)
    route_clf.fit(X_train, y_train_route, sample_weight=sw)
    joblib.dump(route_clf, OUT_DIR / "route_clf.joblib")

    majority_label = y_train_route.mode().iat[0]
    lines.append("## Route classification (majority-class balanced weighting)\n")
    lines.append(f"Majority-class baseline (always predict `{majority_label}`), on test:\n")
    lines.append("```\n" + majority_baseline_report(y_test_route, majority_label) + "```\n")
    lines.append("Trained HistGradientBoostingClassifier, on val:\n")
    lines.append("```\n" + classification_report(y_val_route, route_clf.predict(X_val), zero_division=0) + "```\n")
    lines.append("Trained HistGradientBoostingClassifier, on test:\n")
    lines.append("```\n" + classification_report(y_test_route, route_clf.predict(X_test), zero_division=0) + "```\n")

    # ---- 2. insulin-type classifier, conditional on a dose being given ----
    dosed_train = train[train["route"] != "NONE"]
    dosed_val = val[val["route"] != "NONE"]
    dosed_test = test[test["route"] != "NONE"]

    Xd_train, y_train_type = dosed_train[STATE_COLS], dosed_train["ins_type"]
    Xd_val, y_val_type = dosed_val[STATE_COLS], dosed_val["ins_type"]
    Xd_test, y_test_type = dosed_test[STATE_COLS], dosed_test["ins_type"]

    sw_type = compute_sample_weight("balanced", y_train_type)
    type_clf = HistGradientBoostingClassifier(random_state=0, max_iter=300)
    type_clf.fit(Xd_train, y_train_type, sample_weight=sw_type)
    joblib.dump(type_clf, OUT_DIR / "ins_type_clf.joblib")

    lines.append("## Insulin-type classification (rows with route != NONE only)\n")
    lines.append(f"n_dosed: train={len(dosed_train)} val={len(dosed_val)} test={len(dosed_test)}\n")
    lines.append("Trained HistGradientBoostingClassifier, on test:\n")
    lines.append("```\n" + classification_report(y_test_type, type_clf.predict(Xd_test), zero_division=0) + "```\n")

    # ---- 3. dose regressor, conditional on a dose being given (log1p target) ----
    y_train_dose = np.log1p(dosed_train["dose"])
    y_val_dose = np.log1p(dosed_val["dose"])
    y_test_dose = np.log1p(dosed_test["dose"])

    dose_reg = HistGradientBoostingRegressor(random_state=0, max_iter=300)
    dose_reg.fit(Xd_train, y_train_dose)
    joblib.dump(dose_reg, OUT_DIR / "dose_reg.joblib")

    pred_val_dose = np.expm1(dose_reg.predict(Xd_val))
    pred_test_dose = np.expm1(dose_reg.predict(Xd_test))
    true_val_dose = dosed_val["dose"].to_numpy()
    true_test_dose = dosed_test["dose"].to_numpy()

    naive_pred = np.full(len(true_test_dose), dosed_train["dose"].median())

    lines.append("## Dose regression (units; trained on log1p(dose), rows with route != NONE)\n")
    lines.append(f"val:  MAE={mean_absolute_error(true_val_dose, pred_val_dose):.2f}  "
                  f"RMSE={mean_squared_error(true_val_dose, pred_val_dose) ** 0.5:.2f}\n")
    lines.append(f"test: MAE={mean_absolute_error(true_test_dose, pred_test_dose):.2f}  "
                  f"RMSE={mean_squared_error(true_test_dose, pred_test_dose) ** 0.5:.2f}\n")
    lines.append(f"naive median-dose baseline, test: MAE={mean_absolute_error(true_test_dose, naive_pred):.2f}  "
                  f"RMSE={mean_squared_error(true_test_dose, naive_pred) ** 0.5:.2f}\n")

    # ---- feature importance (permutation-free, HGB's built-in) ----
    lines.append("## Feature list used for all three models\n")
    lines.append(f"{STATE_COLS}\n")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote models to {OUT_DIR}")
    print(f"wrote report to {REPORT}")


if __name__ == "__main__":
    main()
