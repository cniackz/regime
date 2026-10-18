#!/usr/bin/env python3
"""Mutation testing for regime.go with the standard Go toolchain only.

Each mutant is one literal replacement in regime.go. The script applies it,
runs `go test -count=1 .` (the bank stays skipped), records whether some test
failed (killed) or all passed (survived), and restores the file. Run from the
repository root:

    python3 scripts/mutants.py
"""
import pathlib
import subprocess
import sys

MUTANTS = [
    ("hazard: change term uses 1-H", "nr.logp = d.logH +", "nr.logp = d.log1mH +"),
    ("hazard: growth term uses H", "+ d.log1mH\n", "+ d.logH\n"),
    ("predictive: scale inverted", "scale2 = r.beta / r.alpha * (1 + q)", "scale2 = r.alpha / r.beta * (1 + q)"),
    ("predictive: dof halved", "nu = 2 * r.alpha", "nu = r.alpha"),
    ("predictive: Student-t exponent", "(nu+1)/2*math.Log1p", "nu/2*math.Log1p"),
    ("update: beta step doubled", "r.beta += e * e / (2 * s)", "r.beta += e * e / s"),
    ("update: slope sign", "r.m[1] += k1 * e", "r.m[1] -= k1 * e"),
    ("update: level gain dropped", "r.m[0] += k0 * e", "r.m[0] += 0 * e"),
    ("update: V shrink halved", "r.v[1][1] -= k1 * vx1", "r.v[1][1] -= 0.5 * k1 * vx1"),
    ("prune: keep only the max", "if rs[i].logp-mx >= pruneLog {", "if rs[i].logp-mx >= 0 {"),
    ("prune: drop the most probable", "if rs[i].logp < rs[lo].logp {", "if rs[i].logp > rs[lo].logp {"),
    ("changeProb: window strict", "r.n <= d.cfg.window", "r.n < d.cfg.window"),
    ("changeProb: count the initial regime", "if !r.initial && r.n", "if r.n"),
    ("alarm: strict threshold", "if d.armed && cp >= d.cfg.threshold {", "if d.armed && cp > d.cfg.threshold {"),
    ("alarm: never disarm", "\t\talarm = true\n\t\td.armed = false", "\t\talarm = true\n\t\td.armed = true"),
    ("debounce: rearm at once", "d.n-d.lastAlarm >= d.cfg.minRun", "true"),
    ("debounce: count from the most probable regime", "d.n-d.lastAlarm >= d.cfg.minRun", "d.runs[d.mapIndex()].n >= d.cfg.minRun"),
    ("debounce: forget the alarm time", "\t\td.lastAlarm = d.n\n", "\n"),
    ("range: maxStd 1e200", "const maxStd = 1e100", "const maxStd = 1e200"),
    ("prune: threshold 1e-10", "var pruneLog = math.Log(1e-12)", "var pruneLog = math.Log(1e-10)"),
    ("clamp: q in predict", "q := max(x[0]*vx0+x[1]*vx1, 0)", "q := x[0]*vx0 + x[1]*vx1"),
    ("clamp: s in update", "s := 1 + max(x[0]*vx0+x[1]*vx1, 0)", "s := 1 + x[0]*vx0 + x[1]*vx1"),
    ("clamp: V00 floor", "\tr.v[0][0] = max(r.v[0][0], 0)\n", ""),
    ("clamp: V11 floor", "\tr.v[1][1] = max(r.v[1][1], 0)\n", ""),
    ("clamp: SlopeStd floor", "math.Sqrt(max(r.beta/(r.alpha-1)*r.v[1][1], 0))", "math.Sqrt(r.beta / (r.alpha - 1) * r.v[1][1])"),
    ("center: unweighted sum", "center += r.p * loc", "center += loc"),
    ("since: low quantile at the median", "cum >= 0.05", "cum >= 0.5"),
    ("units: MAD constant", "scale := 1.4826 * median(s)", "scale := median(s)"),
    ("units: time unit ignored in step", "ut := (t - d.t0) / d.dt0\n\tif !inRange(y)", "ut := (t - d.t0)\n\tif !inRange(y)"),
    ("report: slope time unit", "rep.Slope = d.s0 / d.dt0 * r.m[1]", "rep.Slope = d.s0 * r.m[1]"),
    ("report: level not unscaled", "Level:      d.m0 + d.s0*lvl,", "Level:      d.m0 + lvl,"),
    ("guard: skip the finite check", "if !allFinite(next) {", "if false {"),
    ("guard: time order", "if d.n > 0 && !(t > d.lastT) {", "if d.n > 0 && t < d.lastT {"),
]

def main():
    path = pathlib.Path("regime.go")
    orig = path.read_text()
    killed, survived = [], []
    for name, old, new in MUTANTS:
        if orig.count(old) != 1:
            print(f"BAD MUTANT (matches {orig.count(old)} times): {name}")
            sys.exit(2)
        path.write_text(orig.replace(old, new))
        try:
            r = subprocess.run(["go", "test", "-count=1", "-timeout", "300s", "."],
                               capture_output=True, text=True)
        finally:
            path.write_text(orig)
        (killed if r.returncode != 0 else survived).append(name)
        print(("KILLED   " if r.returncode != 0 else "SURVIVED ") + name, flush=True)
    print(f"\n{len(killed)}/{len(MUTANTS)} killed")
    if survived:
        print("survivors:\n  " + "\n  ".join(survived))
        sys.exit(1)

if __name__ == "__main__":
    main()
