# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +6.1689 | [+5.9884, +6.3332] |
| FQE | +7.1879 | [+7.0210, +7.3474] |
| WIS | +13.6921 | [+12.0112, +14.4593] |
| WDR | +7.5943 | [+6.5182, +8.8390] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +6.1689. The learned policy's WDR estimate is +7.5943 [+6.5182, +8.8390], whose 95% CI is **ABOVE** the clinician point estimate. FQE agrees at +7.1879 [+7.0210, +7.3474].

Learned policy agreement with clinician actions (greedy): 0.705

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
