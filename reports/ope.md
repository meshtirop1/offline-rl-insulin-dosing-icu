# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +5.5396 | [+5.3535, +5.7240] |
| FQE | +6.3854 | [+6.2239, +6.5360] |
| WIS | +12.2502 | [+10.4842, +13.0925] |
| WDR | +6.2412 | [+4.2218, +8.0140] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +5.5396. The learned policy's WDR estimate is +6.2412 [+4.2218, +8.0140], whose 95% CI is **OVERLAPS** the clinician point estimate. FQE agrees at +6.3854 [+6.2239, +6.5360].

Learned policy agreement with clinician actions (greedy): 0.700

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
