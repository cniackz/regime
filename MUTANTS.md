# Mutation testing

Run with `python3 scripts/mutants.py`. Each line is one literal change to `regime.go`; KILLED means at least one test failed.

```
KILLED   hazard: change term uses 1-H
KILLED   hazard: growth term uses H
KILLED   predictive: scale inverted
KILLED   predictive: dof halved
KILLED   predictive: Student-t exponent
KILLED   update: beta step doubled
KILLED   update: slope sign
KILLED   update: level gain dropped
KILLED   update: V shrink halved
KILLED   prune: keep only the max
KILLED   prune: drop the most probable
KILLED   changeProb: window strict
KILLED   changeProb: count the initial regime
KILLED   alarm: strict threshold
KILLED   alarm: never disarm
KILLED   debounce: rearm at once
KILLED   center: unweighted sum
KILLED   since: low quantile at the median
KILLED   units: MAD constant
KILLED   units: time unit ignored in step
KILLED   report: slope time unit
KILLED   report: level not unscaled
KILLED   guard: skip the finite check
KILLED   guard: time order

24/24 killed
```
