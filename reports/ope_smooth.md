# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +2.4267 | [+2.2516, +2.5968] |
| FQE | +2.8337 | [+2.7323, +2.9341] |
| WIS | +3.3469 | [+2.2873, +5.4047] |
| WDR | +1.4442 | [-0.3021, +2.6844] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +2.4267. The learned policy's WDR estimate is +1.4442 [-0.3021, +2.6844], whose 95% CI is **OVERLAPS** the clinician point estimate. FQE agrees at +2.8337 [+2.7323, +2.9341].

Learned policy agreement with clinician actions (greedy): 0.705

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
