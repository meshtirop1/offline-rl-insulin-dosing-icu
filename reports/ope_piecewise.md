# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +5.5396 | [+5.3535, +5.7240] |
| FQE | +6.5506 | [+6.3959, +6.7105] |
| WIS | +13.4826 | [+11.4054, +14.2025] |
| WDR | +7.8633 | [+6.0150, +9.2660] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +5.5396. The learned policy's WDR estimate is +7.8633 [+6.0150, +9.2660], whose 95% CI is **ABOVE** the clinician point estimate. FQE agrees at +6.5506 [+6.3959, +6.7105].

Learned policy agreement with clinician actions (greedy): 0.705

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
