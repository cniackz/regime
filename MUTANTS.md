# Mutation testing

Run with `python3 scripts/mutants.py`. Each line is one literal change to `regime.go`; KILLED means at least one test failed.

Raw count: 33 of 33 killed. Declared equivalent: none. Adjusted count: 33 of 33.

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
KILLED   debounce: count from the most probable regime
KILLED   debounce: forget the alarm time
KILLED   range: maxStd 1e200
KILLED   prune: threshold 1e-10
KILLED   clamp: q in predict
KILLED   clamp: s in update
KILLED   clamp: V00 floor
KILLED   clamp: V11 floor
KILLED   clamp: SlopeStd floor
KILLED   center: unweighted sum
KILLED   since: low quantile at the median
KILLED   units: MAD constant
KILLED   units: time unit ignored in step
KILLED   report: slope time unit
KILLED   report: level not unscaled
KILLED   guard: skip the finite check
KILLED   guard: time order

33/33 killed
```
