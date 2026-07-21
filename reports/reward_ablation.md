# Reward-sensitivity ablation

Each row retrains CQL from scratch under a different reward, then re-runs OPE.
A trustworthy conclusion has the SAME direction across all variants.

| reward variant | clinician V | WDR | WDR 95% CI | WDR vs clinician |
|---|---|---|---|---|
| piecewise | +5.5396 | +3.0742 | [+2.0353, +4.7953] | BELOW |
| tir_binary | +6.1689 | +5.2609 | [+3.9566, +6.6199] | OVERLAPS |
| asymmetric | +5.4827 | +5.5145 | [+4.2245, +6.8613] | OVERLAPS |
| smooth | +2.4267 | +1.4442 | [-0.3021, +2.6844] | OVERLAPS |

**Verdict: UNSTABLE -- direction flips across reward variants; do NOT report a benefit claim**
