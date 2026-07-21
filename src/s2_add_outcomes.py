"""
Link MIMIC-III outcome/demographic tables onto the episode table.

Adds, per ICU stay:
  - care_unit            : FIRST_CAREUNIT (MICU/SICU/CSRU/TSICU/CCU) -- state feature;
                           different units dose insulin very differently.
  - age                  : years at ICU admission (MIMIC shifts ages >89 to ~300;
                           we cap at 90 per the standard MIMIC convention).
  - gender               : M/F -> 1/0.
  - hospital_expire_flag : in-hospital mortality (0/1), the real outcome.
  - terminal_mortality_reward : 0 everywhere except the LAST bin of a stay, where it
                           is +1 if the patient survived to discharge and -1 if they
                           died in hospital. This lets OPE score a policy against a
                           real outcome instead of only the glucose proxy.

IMPORTANT: mortality is an admission-level outcome driven by many factors; insulin
dosing is one small input. The terminal-mortality reward is provided as a SECONDARY,
exploratory evaluation signal (does a glucose-improving policy at least not track
with worse mortality?), NOT as evidence that the policy causally changes mortality.
Do not train a policy to directly maximize this and claim lives saved -- that is the
exact overreach the sepsis AI-Clinician line of work was criticized for.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

EPISODES = cfg.EPISODES
OUT = cfg.EPISODES_ENRICHED


def main():
    ep = pd.read_parquet(EPISODES)

    icu = pd.read_csv(cfg.ICUSTAYS_CSV,
                      usecols=["ICUSTAY_ID", "HADM_ID", "SUBJECT_ID", "FIRST_CAREUNIT", "INTIME"])
    adm = pd.read_csv(cfg.ADMISSIONS_CSV,
                      usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"])
    pat = pd.read_csv(cfg.PATIENTS_CSV, usecols=["SUBJECT_ID", "GENDER", "DOB"])

    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")
    pat["DOB"] = pd.to_datetime(pat["DOB"], errors="coerce")

    # age at ICU admission; MIMIC shifts ages >89 to ~300y -> cap at 90
    icu = icu.merge(pat, on="SUBJECT_ID", how="left")
    age = (icu["INTIME"].dt.year - icu["DOB"].dt.year)
    icu["age"] = np.where(age > 89, 90, age).astype(float)
    icu["gender"] = (icu["GENDER"] == "M").astype(int)

    icu = icu.merge(adm, on="HADM_ID", how="left")

    stay = icu[["ICUSTAY_ID", "FIRST_CAREUNIT", "age", "gender",
                "HOSPITAL_EXPIRE_FLAG"]].rename(
        columns={"FIRST_CAREUNIT": "care_unit", "HOSPITAL_EXPIRE_FLAG": "hospital_expire_flag"})

    n_before = len(ep)
    ep = ep.merge(stay, on="ICUSTAY_ID", how="left")
    assert len(ep) == n_before, "merge changed row count"

    # terminal mortality reward on the last bin of each stay
    ep["terminal_mortality_reward"] = 0.0
    last_mask = ep["done"] == True
    ep.loc[last_mask, "terminal_mortality_reward"] = np.where(
        ep.loc[last_mask, "hospital_expire_flag"] == 1, -1.0, 1.0)

    # median-impute the handful of missing ages, if any
    ep["age"] = ep["age"].fillna(ep["age"].median())
    ep["gender"] = ep["gender"].fillna(0).astype(int)
    ep["care_unit"] = ep["care_unit"].fillna("UNK")

    ep.to_parquet(OUT, index=False)

    print(f"enriched episodes: {len(ep)} rows -> {OUT}")
    print(f"missing outcome (unmatched stays): {ep['hospital_expire_flag'].isna().sum()}")
    print("\ncare_unit distribution (bins):")
    print(ep["care_unit"].value_counts().to_dict())
    print(f"\nage: median={ep['age'].median():.0f} p5={ep['age'].quantile(.05):.0f} "
          f"p95={ep['age'].quantile(.95):.0f}")
    print(f"gender (frac M): {ep['gender'].mean():.3f}")
    died = ep.loc[last_mask, "hospital_expire_flag"]
    print(f"\nstay-level in-hospital mortality: {100*died.mean():.2f}% of {len(died)} stays")
    print("terminal_mortality_reward on last bins:")
    print(ep.loc[last_mask, "terminal_mortality_reward"].value_counts().to_dict())


if __name__ == "__main__":
    main()
