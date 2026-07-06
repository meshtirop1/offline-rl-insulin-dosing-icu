"""
Build an offline-RL episode dataset from glucose_insulin_pair.csv.

MDP formulation
----------------
Time is discretized into fixed-width bins (default 4h, matching the AI Clinician /
GLUCOSE convention) per ICUSTAY_ID. Within each bin:

  STATE (observed at start of bin, using only information available up to that point):
    - last_glc              : most recent glucose value carried forward (LOCF)
    - glc_slope              : (last_glc - glc two readings ago) / dt, trend signal
    - n_glc_readings_in_bin  : glucose measurement density (informative: sicker/labile
                                patients get checked more often)
    - hours_since_last_glc   : recency of information
    - hours_since_last_dose  : time since any insulin was last given
    - dose_last_24h          : cumulative insulin (units) in the trailing 24h, by route
    - los_icu_days, first_icu_stay : static admission context
    - bin_index              : position in the stay (proxy for acuity trajectory / day of stay)

  ACTION (hybrid discrete+continuous, applied during the bin):
    - route     : categorical  {NONE, BOLUS_INYECTION, BOLUS_PUSH, INFUSION}
    - ins_type  : categorical  {NONE, Short, Intermediate, Long}
    - dose      : continuous units (0 if route == NONE)
  This factored action space is the point of departure from prior work (GLUCOSE,
  Insulin4RL), which model IV-infusion-rate-only, single-route control.

  REWARD: computed from the glucose observed in the *following* bin (i.e. the
  consequence of the action just taken), using a piecewise clinical scale:
    glc < 40           -> -1.0   (severe hypoglycemia, primary safety failure)
    40 <= glc < 70      -> -0.4   (hypoglycemia)
    70 <= glc <= 180     -> +1.0   (target range)
    180 < glc <= 300     -> -0.3   (hyperglycemia)
    glc > 300           -> -0.6   (severe hyperglycemia)
  Reward is 0 for bins with no new glucose observation (no signal that step).
  NOTE: this reward is a first pass for prototyping the pipeline; it should be
  ablated/sensitivity-tested before drawing conclusions (see README "Reward
  sensitivity" section) -- reward misspecification is the most common critique
  of AI-Clinician-style ICU RL papers.

Output
------
Parquet file with one row per (ICUSTAY_ID, bin_index): state columns, action
columns, reward, done flag, plus SUBJECT_ID for patient-level splitting.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BIN_HOURS = 4

STATE_COLS = [
    "last_glc", "glc_slope", "n_glc_readings_in_bin", "hours_since_last_glc",
    "hours_since_last_dose", "dose_last_24h", "basal_long_24h",
    "basal_intermediate_24h", "on_basal", "los_icu_days", "first_icu_stay",
    "bin_index",
]
ROUTES = ["NONE", "BOLUS_INYECTION", "BOLUS_PUSH", "INFUSION"]
INS_TYPES = ["NONE", "Short", "Intermediate", "Long"]


def reward_from_glc(glc: float) -> float:
    if pd.isna(glc):
        return 0.0
    if glc < 40:
        return -1.0
    if glc < 70:
        return -0.4
    if glc <= 180:
        return 1.0
    if glc <= 300:
        return -0.3
    return -0.6


def load_raw(path: Path) -> pd.DataFrame:
    usecols = [
        "SUBJECT_ID", "ICUSTAY_ID", "LOS_ICU_days", "first_ICU_stay",
        "TIMER", "INPUT", "INSULINTYPE", "EVENT", "GLC",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["TIMER"] = pd.to_datetime(df["TIMER"], utc=True, errors="coerce")
    df = df.dropna(subset=["TIMER", "ICUSTAY_ID"])
    df["first_ICU_stay"] = (
        df["first_ICU_stay"].astype(str).str.lower().map({"true": 1, "false": 0})
    )
    return df


def _mode_str(values):
    """Most common string in a small list (ties -> first-seen order via Counter)."""
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def build_episodes(df: pd.DataFrame, bin_hours: int = BIN_HOURS) -> pd.DataFrame:
    """Vectorized-per-stay episode builder.

    Semantics identical to the original per-bin implementation, but instead of
    re-filtering the stay's DataFrame once per 4h bin (O(bins x rows), the prior
    bottleneck), each event is bucketed into its bin once with numpy and the bin
    loop reads pre-grouped index lists. ~50x faster on the full cohort.
    """
    rows = []
    bin_ns = int(bin_hours * 3.6e12)      # bin width in nanoseconds
    win24_ns = int(24 * 3.6e12)

    for icustay_id, g in df.groupby("ICUSTAY_ID", sort=False):
        g = g.sort_values("TIMER")
        subject_id = g["SUBJECT_ID"].iloc[0]
        los = g["LOS_ICU_days"].iloc[0]
        first_stay = g["first_ICU_stay"].iloc[0]

        # force nanosecond resolution: this pandas build stores datetime as us,
        # so a plain .astype("int64") would return microseconds and silently
        # break the bin-width arithmetic.
        timer_ns = g["TIMER"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
        glc = g["GLC"].to_numpy(dtype=float)
        event = g["EVENT"].to_numpy(dtype=object)
        input_units = g["INPUT"].to_numpy(dtype=float)
        instype = g["INSULINTYPE"].to_numpy(dtype=object)

        t0_ns = (timer_ns.min() // bin_ns) * bin_ns
        t_end_ns = -(-timer_ns.max() // bin_ns) * bin_ns   # ceil division
        n_bins = max(1, int((t_end_ns - t0_ns) // bin_ns))
        bin_idx = ((timer_ns - t0_ns) // bin_ns).astype(int)

        # bucket event indices by bin (single pass)
        buckets = [[] for _ in range(n_bins)]
        for i, b in enumerate(bin_idx):
            if 0 <= b < n_bins:
                buckets[b].append(i)

        last_glc = np.nan
        prev_glc = np.nan
        last_glc_time = None      # ns
        last_dose_time = None     # ns
        dose_history = []         # (time_ns, units)
        basal_history = []        # (time_ns, long_units, inter_units)

        for b in range(n_bins):
            bin_start_ns = t0_ns + b * bin_ns
            idxs = buckets[b]

            glc_i = [i for i in idxs if not np.isnan(glc[i])]
            dose_i = [i for i in idxs if event[i] is not None and not (isinstance(event[i], float) and np.isnan(event[i]))]

            hours_since_last_glc = (bin_start_ns - last_glc_time) / 3.6e12 if last_glc_time is not None else np.nan
            hours_since_last_dose = (bin_start_ns - last_dose_time) / 3.6e12 if last_dose_time is not None else np.nan

            dose_history = [(t, u) for t, u in dose_history if (bin_start_ns - t) <= win24_ns]
            dose_last_24h = sum(u for _, u in dose_history)
            basal_history = [(t, l, i) for t, l, i in basal_history if (bin_start_ns - t) <= win24_ns]
            basal_long_24h = sum(l for _, l, _ in basal_history)
            basal_intermediate_24h = sum(i for _, _, i in basal_history)

            state = {
                "last_glc": last_glc,
                "glc_slope": (last_glc - prev_glc) / bin_hours if not pd.isna(prev_glc) else 0.0,
                "n_glc_readings_in_bin": len(glc_i),
                "hours_since_last_glc": hours_since_last_glc,
                "hours_since_last_dose": hours_since_last_dose,
                "dose_last_24h": dose_last_24h,
                "basal_long_24h": basal_long_24h,
                "basal_intermediate_24h": basal_intermediate_24h,
                "on_basal": 1 if (basal_long_24h + basal_intermediate_24h) > 0 else 0,
                "los_icu_days": los,
                "first_icu_stay": first_stay,
                "bin_index": b,
            }

            if dose_i:
                route = _mode_str([event[i] for i in dose_i])
                types = [instype[i] for i in dose_i if isinstance(instype[i], str)]
                ins_type = _mode_str(types) if types else "NONE"
                dose = float(np.nansum([input_units[i] for i in dose_i]))
                last_dose_time = max(timer_ns[i] for i in dose_i)
                dose_history.append((last_dose_time, dose))
            else:
                route, ins_type, dose = "NONE", "NONE", 0.0

            # reactive action = short-acting correction (SC bolus / SC push / IV infusion)
            reactive_i = [i for i in dose_i if instype[i] == "Short"]
            if reactive_i:
                reactive_route = _mode_str([event[i] for i in reactive_i])
                reactive_dose = float(np.nansum([input_units[i] for i in reactive_i]))
            else:
                reactive_route, reactive_dose = "NONE", 0.0

            # record basal given THIS bin for future trailing-24h windows
            long_units = float(np.nansum([input_units[i] for i in dose_i if instype[i] == "Long"]))
            inter_units = float(np.nansum([input_units[i] for i in dose_i if instype[i] == "Intermediate"]))
            if long_units > 0 or inter_units > 0:
                basal_history.append((bin_start_ns + bin_ns, long_units, inter_units))

            # next-bin glucose determines reward (consequence of this bin's action)
            next_glc = np.nan
            if b + 1 < n_bins:
                nglc_i = [i for i in buckets[b + 1] if not np.isnan(glc[i])]
                if nglc_i:
                    next_glc = float(glc[nglc_i[-1]])
            reward = reward_from_glc(next_glc)

            if glc_i:
                prev_glc = last_glc
                last_glc = float(glc[glc_i[-1]])
                last_glc_time = timer_ns[glc_i[-1]]

            rows.append({
                "SUBJECT_ID": subject_id,
                "ICUSTAY_ID": icustay_id,
                **state,
                "route": route,
                "ins_type": ins_type,
                "dose": dose,
                "reactive_route": reactive_route,
                "reactive_dose": reactive_dose,
                "next_glc": next_glc,
                "reward": reward,
                "done": b == n_bins - 1,
            })

    return pd.DataFrame(rows)


def split_by_subject(episodes: pd.DataFrame, seed: int = 0):
    subjects = episodes["SUBJECT_ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_ids = set(subjects[:n_train])
    val_ids = set(subjects[n_train:n_train + n_val])
    test_ids = set(subjects[n_train + n_val:])
    split = np.select(
        [episodes["SUBJECT_ID"].isin(train_ids), episodes["SUBJECT_ID"].isin(val_ids)],
        ["train", "val"],
        default="test",
    )
    return episodes.assign(split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"C:\Users\mtiro\Downloads\glycemic\glycemic\glucose_insulin_pair.csv")
    ap.add_argument("--out", default=r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\data\processed\episodes.parquet")
    ap.add_argument("--bin-hours", type=int, default=BIN_HOURS)
    ap.add_argument("--limit-icustays", type=int, default=None, help="debug: only process first N ICU stays")
    args = ap.parse_args()

    df = load_raw(Path(args.input))
    if args.limit_icustays:
        keep = df["ICUSTAY_ID"].drop_duplicates().iloc[: args.limit_icustays]
        df = df[df["ICUSTAY_ID"].isin(keep)]

    episodes = build_episodes(df, bin_hours=args.bin_hours)
    episodes = split_by_subject(episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(out_path, index=False)

    print(f"episodes: {len(episodes)} rows, {episodes['ICUSTAY_ID'].nunique()} ICU stays, "
          f"{episodes['SUBJECT_ID'].nunique()} patients")
    print(episodes["split"].value_counts())
    print("\nfull-dose route distribution (all insulin):")
    print(episodes["route"].value_counts())
    print("\nreactive-action route distribution (RL action space):")
    print(episodes["reactive_route"].value_counts())
    print("\non_basal state flag distribution:")
    print(episodes["on_basal"].value_counts())
    print("\nreward distribution (nonzero only):")
    print(episodes.loc[episodes["reward"] != 0, "reward"].value_counts())
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
