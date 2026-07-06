# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +5.4827 | [+5.2956, +5.6640] |
| FQE | +6.4669 | [+6.3123, +6.6232] |
| WIS | +12.7938 | [+11.3108, +13.5973] |
| WDR | +6.9918 | [+5.8791, +8.2991] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +5.4827. The learned policy's WDR estimate is +6.9918 [+5.8791, +8.2991], whose 95% CI is **ABOVE** the clinician point estimate. FQE agrees at +6.4669 [+6.3123, +6.6232].

Learned policy agreement with clinician actions (greedy): 0.703

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
