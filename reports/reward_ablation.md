# Reward-sensitivity ablation

Each row retrains CQL from scratch under a different reward, then re-runs OPE.
A trustworthy conclusion has the SAME direction across all variants.

| reward variant | clinician V | WDR | WDR 95% CI | WDR vs clinician |
|---|---|---|---|---|
| piecewise | +5.5396 | +7.8633 | [+6.0150, +9.2660] | ABOVE |
| tir_binary | +6.1689 | +7.5943 | [+6.5182, +8.8390] | ABOVE |
| asymmetric | +5.4827 | +6.9918 | [+5.8791, +8.2991] | ABOVE |
| smooth | +2.4267 | +3.6415 | [+1.4115, +5.8394] | OVERLAPS |

**Verdict: UNSTABLE -- direction flips across reward variants; do NOT report a benefit claim**
