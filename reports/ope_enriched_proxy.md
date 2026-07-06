# Off-policy evaluation (test split, patient-disjoint)
gamma=0.95 target_softmax_tau=0.5 ratio_clip=+/-5.0 prob_floor=0.001 n_bootstrap=200
test trajectories (ICU stays): 1842

## Estimated discounted return under the PROXY reward
| estimator | value | 95% CI |
|---|---|---|
| clinician_Vpi_b | +5.5396 | [+5.3535, +5.7240] |
| FQE | +5.5846 | [+5.4179, +5.7454] |
| WIS | +8.7477 | [+7.1169, +11.1862] |
| WDR | +3.0742 | [+2.0353, +4.7953] |

## Reading the result
Clinician (logged-policy) value under this proxy reward is +5.5396. The learned policy's WDR estimate is +3.0742 [+2.0353, +4.7953], whose 95% CI is **BELOW** the clinician point estimate. FQE agrees at +5.5846 [+5.4179, +5.7454].

Learned policy agreement with clinician actions (greedy): 0.705

## Caveat (do not skip when writing up)
These returns are under the time-in-range PROXY reward, not a clinical outcome. A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU external validation (see README). Also: WIS is high-variance at this horizon; when FQE and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers can see the spread.
