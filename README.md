# Offline Reinforcement Learning for ICU Insulin Dosing

A reproducible benchmark and off-policy evaluation of offline reinforcement
learning (RL) for insulin dosing in a general intensive care unit (ICU)
population, built from the MIMIC-III critical care database.

**Main result:** under careful evaluation, a learned insulin-dosing policy does
**not** outperform ICU clinicians. It matches clinician value on a glucose
time-in-range proxy (fitted Q evaluation is level with clinicians; the
doubly-robust estimate is slightly below), it is statistically indistinguishable
from clinicians on real in-hospital mortality, and the small apparent proxy-reward
advantage does not hold up when the reward definition is changed. The contribution
is an honest, reusable benchmark rather than a claim that RL beats existing care.

---

## What makes this benchmark different

Most prior offline-RL work on ICU glucose control models a single action, the
intravenous insulin infusion rate, on a narrow cohort. This benchmark instead:

1. **Models the delivery routes actually used at the bedside** — subcutaneous
   bolus, intravenous bolus, and intravenous infusion — over a general
   medical/surgical ICU population.
2. **Treats scheduled basal insulin as state context, not as an action.** Long-
   and intermediate-acting insulin is given on a plan rather than in response to
   each glucose reading, so it belongs in the patient state. Putting it in the
   action space caused an earlier version of the policy to abandon basal coverage,
   which this design removes.
3. **Evaluates honestly.** Four off-policy estimators (clinician value, fitted Q
   evaluation, per-decision weighted importance sampling, weighted doubly robust),
   patient-level bootstrap confidence intervals, a real-outcome check against
   in-hospital mortality, and a reward-sensitivity analysis across four reward
   definitions.

---

## Data access (not included in this repository)

This project uses **credentialed** data that cannot be redistributed. Nothing
derived from individual patients is stored here. To reproduce the study you must
obtain the data yourself:

1. Create a [PhysioNet](https://physionet.org/) account, complete the required
   CITI "Data or Specimens Only Research" training, and sign the data use
   agreement.
2. Download the curated glucose-insulin dataset:
   **Curated Data for Describing Blood Glucose Management in the Intensive Care
   Unit** (v1.0.1), https://doi.org/10.13026/517s-2q57 — this provides
   `glucose_insulin_pair.csv`.
3. Download the three MIMIC-III core tables for outcome linkage:
   `ADMISSIONS.csv`, `PATIENTS.csv`, `ICUSTAYS.csv` from
   **MIMIC-III Clinical Database v1.4**, https://doi.org/10.13026/C2XW26.
4. Place the files where the scripts expect them and update the path constants at
   the top of each script in `src/` (see **Configuration** below).

The dataset is a curated extract of MIMIC-III v1.4 (MetaVision records,
2008–2012): about 9,500 patients, 12,210 ICU stays, and roughly 604,000
glucose/insulin events, with insulin administrations paired to a preceding
glucose reading by clinician-defined timing rules.

---

## Repository layout

The pipeline is eight numbered stages. Each stage reads what the previous one
wrote, so the code can be read straight through in order.

```
run_pipeline.py              Runs the whole study end to end
src/
  config.py                  Every path and hyperparameter, in one place
  models.py                  Q-network shared by stages 5 and 6
  s1_build_episodes.py       Raw events   -> 4-hour decision bins        (Sec. 4.1-4.2)
  s2_add_outcomes.py         Link MIMIC-III mortality and demographics   (Sec. 3)
  s3_prepare_dataset.py      Discretize actions, standardize state       (Sec. 4.2)
  s4_behavior_cloning.py     Sanity check on the state representation    (Sec. 4.4, 5.1)
  s5_train_cql.py            Conservative Q-Learning policy              (Sec. 4.3, 5.2)
  s6_evaluate_ope.py         Four estimators vs clinicians               (Sec. 4.5, 5.3)
  s7_reward_ablation.py      Retrain/re-evaluate under four rewards      (Sec. 4.6, 5.4)
  s8_make_figures.py         Figures used in the paper                   (Figures 1-7)
reports/                     Text result reports (metrics, tables, ablations)
```

No stage imports another, and no path is hardcoded: everything resolves through
`src/config.py`.

---

## Requirements

- Python 3.10 or newer
- See `requirements.txt`. The pipeline runs on CPU; no GPU is required.

```bash
pip install -r requirements.txt
```

---

## Configuration

All paths and hyperparameters live in `src/config.py`. Point the two credentialed
inputs at wherever you downloaded them:

```bash
# Linux / macOS
export GLYCEMIC_RAW_DIR=/path/to/glucose-management-mimic   # holds glucose_insulin_pair.csv
export MIMIC_DIR=/path/to/mimic-iii                         # holds ADMISSIONS/PATIENTS/ICUSTAYS.csv
```

```powershell
# Windows PowerShell
$env:GLYCEMIC_RAW_DIR = "C:\path\to\glucose-management-mimic"
$env:MIMIC_DIR        = "C:\path\to\mimic-iii"
```

If unset, they default to `data/raw` and `data/mimic` inside the project. Every
other path (episodes, tensors, models, reports, figures) is derived automatically.

---

## Reproducing the study

Run everything with one command:

```bash
python run_pipeline.py
```

Useful variants:

```bash
python run_pipeline.py --quick      # smoke test on 200 ICU stays
python run_pipeline.py --from 5     # resume from stage 5
python run_pipeline.py --only 6     # run a single stage
```

Or run stages individually:

```bash
python src/s1_build_episodes.py     # raw events -> 4-hour decision bins
python src/s2_add_outcomes.py       # link mortality, age, sex, care unit
python src/s3_prepare_dataset.py    # discretize actions, write RL tensors
python src/s4_behavior_cloning.py   # sanity check on the state
python src/s5_train_cql.py          # train the policy
python src/s6_evaluate_ope.py       # evaluate against clinicians (proxy reward)
python src/s7_reward_ablation.py    # reward-sensitivity analysis
python src/s8_make_figures.py       # figures
```

To evaluate against real in-hospital mortality instead of the glucose proxy:

```bash
python src/s6_evaluate_ope.py --reward-key mortality_reward \
       --report reports/ope_enriched_mortality.md
```

---

## Method overview

- **Decision process.** Each ICU stay is split into 4-hour bins. The state
  summarizes recent glucose and its trend, insulin already on board, trailing
  24-hour basal exposure, and age, sex, and care unit. The action is the reactive
  short-acting correction: none, or one of three routes crossed with a low, medium,
  or high dose tertile (ten discrete actions). The reward reads the next bin's
  glucose on a time-in-range scale, with a terminal survival signal at discharge.
- **Policy.** Discrete Conservative Q-Learning with a double-DQN target and a
  Polyak-averaged target network.
- **Evaluation.** Off-policy value under a softened target policy, estimated four
  ways, with 200-sample patient bootstrap confidence intervals, against both the
  glucose proxy and in-hospital mortality.

---

## Results (test split, 95% patient-bootstrap intervals)

| Estimator | Glucose proxy | In-hospital mortality |
|---|---|---|
| Clinician (baseline) | +5.54 [5.35, 5.72] | +0.369 [0.353, 0.385] |
| Fitted Q evaluation | +5.58 [5.42, 5.75] | +0.366 [0.356, 0.378] |
| Weighted importance sampling | +8.75 [7.12, 11.19] | +1.32 [0.39, 1.70] |
| Weighted doubly robust | +3.07 [2.04, 4.80] | +0.565 [0.262, 0.705] |

The doubly-robust estimate sits at or below the clinician value on the proxy, and
every mortality interval includes the clinician value. Importance sampling is the
high-variance estimator and is reported for completeness. A reward-sensitivity
analysis (see `reports/reward_ablation.md`) shows the direction of the proxy
comparison is not stable across reward definitions, so no benefit is claimed.

---

## Citing this work

If you use this benchmark, please cite the data sources it depends on:

- Robles Arévalo A, Mateo-Collado R, Celi LA. Curated Data for Describing Blood
  Glucose Management in the Intensive Care Unit (v1.0.1). PhysioNet. 2021.
  https://doi.org/10.13026/517s-2q57
- Robles Arévalo A, Maley JH, Baker L, et al. Data-driven curation process for
  describing the blood glucose management in the intensive care unit. Scientific
  Data. 2021;8:80.
- Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical
  care database. Scientific Data. 2016;3:160035.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and
  PhysioNet. Circulation. 2000;101(23):e215–e220.

---

## Authors

- Tirop Meshack — Department of Smart Computing, Kyungdong University, Goseong,
  South Korea
- Ajax Don Christ Ndikuriyo — Department of Smart Computing, Kyungdong University
- Nshimirimana Albert — Department of International Business Administration,
  Kyungdong University
- Baseem Al-Athwari (corresponding, baseem_cs@v.kduniv.ac.kr) — Department of
  Smart Computing, Kyungdong University

---

## License

Code is released under the MIT License (see `LICENSE`). The MIMIC-III and curated
glucose-insulin data are **not** included and are governed by the PhysioNet
Credentialed Health Data Use Agreement.

---

## Acknowledgements

This work uses the MIMIC-III database and the curated glucose-insulin dataset,
both made available through PhysioNet. We thank the teams behind those resources.
