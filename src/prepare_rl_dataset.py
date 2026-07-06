"""
Turn episodes.parquet into tensors ready for discrete offline RL.

Action space (reactive correction only)
---------------------------------------
Basal insulin (Long / Intermediate) is scheduled, not decided reactively every
4h, so it is encoded in the STATE (basal_long_24h, basal_intermediate_24h,
on_basal) rather than in the action space. The RL policy controls only the
reactive short-acting correction, over 3 delivery routes:

    action_id 0            -> NONE (no reactive correction this bin)
    action_id 1..3          -> BOLUS_INYECTION (SC bolus), Short, dose {low,med,high}
    action_id 4..6          -> BOLUS_PUSH, Short, dose {low,med,high}
    action_id 7..9          -> INFUSION (IV), Short, dose {low,med,high}

10 discrete actions, every one of which has thousands of supporting transitions
(no rare-class collapse). Dose tertile edges are fit on TRAIN only.

Rationale: the earlier 16-action version (which put Long/Intermediate insulin in
the action space) produced a CQL policy that abandoned basal coverage -- a
conservatism artifact driven by the <1000-sample rare regimens. Separating
scheduled basal (context) from reactive correction (action) is both more
clinically faithful and removes that failure mode at the source.

State
-----
12 raw features (incl. basal context), plus 3 missing-indicator flags for the
columns with NaNs (last_glc, hours_since_last_glc, hours_since_last_dose --
missing only before the first glucose reading / first dose of a stay). NaNs are
median-imputed using TRAIN statistics, then all features standardized with TRAIN
mean/std.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_STATE_COLS = [
    "last_glc", "glc_slope", "n_glc_readings_in_bin", "hours_since_last_glc",
    "hours_since_last_dose", "dose_last_24h", "basal_long_24h",
    "basal_intermediate_24h", "on_basal", "los_icu_days", "first_icu_stay",
    "bin_index",
    # demographic / acuity context from MIMIC-III linkage (add_outcomes.py)
    "age", "gender",
]
NAN_COLS = ["last_glc", "hours_since_last_glc", "hours_since_last_dose"]
# care_unit one-hot columns added dynamically if present
CARE_UNITS = ["MICU", "SICU", "CSRU", "TSICU", "CCU"]

BASE = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing")
# prefer the outcome-enriched episodes if add_outcomes.py has been run
DATA = BASE / "data" / "processed" / "episodes_enriched.parquet"
if not DATA.exists():
    DATA = BASE / "data" / "processed" / "episodes.parquet"
OUT_DIR = BASE / "data" / "rl"

# reactive delivery routes (all short-acting); order defines action-id blocks
REACTIVE_ROUTES = ["BOLUS_INYECTION", "BOLUS_PUSH", "INFUSION"]


def build_action_table(train_df: pd.DataFrame):
    edges = {}
    action_meta = [{"action_id": 0, "route": "NONE", "dose_bin": -1}]
    next_id = 1
    for route in REACTIVE_ROUTES:
        doses = train_df.loc[train_df["reactive_route"] == route, "reactive_dose"]
        q = doses.quantile([1 / 3, 2 / 3]).to_numpy()
        edges[route] = q
        for b in range(3):
            action_meta.append({"action_id": next_id, "route": route, "dose_bin": b})
            next_id += 1
    return edges, action_meta


def assign_action_id(df: pd.DataFrame, edges: dict) -> np.ndarray:
    action_id = np.zeros(len(df), dtype=np.int64)
    base_id = {route: 1 + i * 3 for i, route in enumerate(REACTIVE_ROUTES)}
    for route, q in edges.items():
        mask = (df["reactive_route"] == route)
        dose = df.loc[mask, "reactive_dose"].to_numpy()
        bin_idx = np.digitize(dose, q)  # 0,1,2
        action_id[mask.to_numpy()] = base_id[route] + bin_idx
    return action_id


def build_state_matrix(df, raw_cols, medians, mean, std):
    flags = pd.DataFrame({f"{c}_missing": df[c].isna().astype(np.float32) for c in NAN_COLS})
    raw = df[raw_cols].fillna(medians)
    X = pd.concat([raw, flags], axis=1).to_numpy(dtype=np.float32)
    return (X - mean) / std


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["ICUSTAY_ID", "bin_index"]).reset_index(drop=True)

    # assemble the raw feature list: drop enriched cols that aren't present, and
    # add care-unit one-hots if the enriched file supplied care_unit
    raw_cols = [c for c in RAW_STATE_COLS if c in df.columns]
    if "care_unit" in df.columns:
        for cu in CARE_UNITS:
            df[f"cu_{cu}"] = (df["care_unit"] == cu).astype(np.float32)
            raw_cols.append(f"cu_{cu}")
    has_mortality = "terminal_mortality_reward" in df.columns
    print(f"using {DATA.name}; enriched state cols: "
          f"{[c for c in raw_cols if c in ('age','gender') or c.startswith('cu_')]}")

    train_df = df[df["split"] == "train"]

    edges, action_meta = build_action_table(train_df)
    df["action_id"] = assign_action_id(df, edges)

    medians = train_df[raw_cols].median()
    tmp = build_state_matrix(train_df, raw_cols, medians, mean=0.0, std=1.0)
    mean = tmp.mean(axis=0)
    std = tmp.std(axis=0)
    std[std == 0] = 1.0
    feature_names = raw_cols + [f"{c}_missing" for c in NAN_COLS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].sort_values(["ICUSTAY_ID", "bin_index"]).reset_index(drop=True)
        X = build_state_matrix(sub, raw_cols, medians, mean, std)
        next_X = np.zeros_like(X)
        next_X[:-1] = X[1:]
        same_stay = (sub["ICUSTAY_ID"].to_numpy()[:-1] == sub["ICUSTAY_ID"].to_numpy()[1:])
        done = sub["done"].to_numpy().astype(np.float32)
        forced_done = np.ones(len(sub), dtype=bool)
        forced_done[:-1] = ~same_stay
        done = np.maximum(done, forced_done.astype(np.float32))

        arrays = dict(
            state=X, action=sub["action_id"].to_numpy(dtype=np.int64),
            reward=sub["reward"].to_numpy(dtype=np.float32),
            next_state=next_X, done=done,
            next_glc=sub["next_glc"].to_numpy(dtype=np.float32),
            subject_id=sub["SUBJECT_ID"].to_numpy(),
            icustay_id=sub["ICUSTAY_ID"].to_numpy(),
        )
        if has_mortality:
            arrays["mortality_reward"] = sub["terminal_mortality_reward"].to_numpy(dtype=np.float32)
            arrays["hospital_expire_flag"] = sub["hospital_expire_flag"].to_numpy(dtype=np.float32)
        np.savez(OUT_DIR / f"{split}.npz", **arrays)
        print(f"{split}: {len(sub)} transitions, {sub['ICUSTAY_ID'].nunique()} episodes")

    meta = {
        "feature_names": feature_names,
        "mean": mean.tolist(), "std": std.tolist(),
        "action_meta": action_meta, "n_actions": len(action_meta),
        "dose_edges": {k: v.tolist() for k, v in edges.items()},
        "gamma": 0.95, "has_mortality": has_mortality,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\naction distribution (train):")
    print(df.loc[train_df.index, "action_id"].value_counts().sort_index())
    print(f"\nn_actions = {len(action_meta)}\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
