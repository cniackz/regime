# regime

Online changepoint detection for noisy series, in Go, with only the standard
library.

You feed a series one sample at a time. After every sample, `regime` tells you:

- how likely it is that a new regime started in the last few samples, and one
  alarm per detected change;
- when the current regime probably started, with a 90% credible interval;
- the current regime's level and slope.

A regime is a stretch of the series with one level (model `Level`) or one level
and one slope (model `Trend`, the default), plus Gaussian noise of unknown
variance.

```go
d, err := regime.New() // or: var d regime.Detector
if err != nil {
	return err
}
for _, s := range samples {
	r, err := d.Observe(s.T, s.V)
	if err != nil {
		continue // NaN, infinity or a time that does not increase; d is unchanged
	}
	if r.Alarm {
		fmt.Printf("new regime since %v (90%%: %v to %v), level %v, slope %v\n",
			r.Since, r.SinceLow, r.SinceHigh, r.Level, r.Slope)
	}
}
```

## How it works

`regime` is Bayesian online changepoint detection (Adams and MacKay 2007), with
the change placed just before a sample (Fearnhead and Liu 2007). It keeps a
posterior over the run length, the number of samples since the last change.
Each run hypothesis carries a conjugate Normal-Inverse-Gamma model of its
regime, so every update is closed form. The package godoc has every formula
and its source.

- **Change probability:** `ChangeProb` is the probability that the current
  regime started within the last `Window` samples. An alarm fires when it
  reaches `Threshold`, then waits `MinRun` samples (the debounce) before it can
  fire again.
- **Units do not matter:** the first `BurnIn` samples fix a location, a scale
  and a time unit (median, median absolute deviation, median spacing). Scaling
  or shifting `v` or `t` gives the same detections.
- **Fixed cost:** at most `MaxRun` hypotheses, so `Observe` does a bounded
  amount of work and does not allocate.

## Options

| Option | Default | Effect |
|---|---|---|
| `WithModel` | `Trend` | `Trend` (level and slope) or `Level` (level only) |
| `WithExpectedRun` | 250 | Prior mean regime length in samples. Larger: fewer false alarms, longer delay |
| `WithThreshold` | 0.5 | `ChangeProb` that raises an alarm |
| `WithWindow` | 10 | How recent, in samples, a start must be to count in `ChangeProb` |
| `WithBurnIn` | 30 | Samples that fix the units. No alarm during burn-in |
| `WithMinRun` | 10 | Debounce in samples after an alarm |
| `WithMaxRun` | 512 | Hypotheses kept: the bound on work per sample |

## False alarms and delay

No detector is both instant and silent. For a known level shift of `delta`
noise standard deviations, any detector whose average run to a false alarm is
`ARL0` samples needs on average about `log(ARL0) / (delta^2 / 2)` samples to
react (Lorden 1971). At `ARL0 = 1000` and `delta = 2` that floor is 3.5
samples.

The synthetic bank in `bank_test.go` measures false alarms per 1000 samples
and the delay at several change sizes, against a two-sided CUSUM. Its claims
were sealed before the first measurement, and they are reported below as they
came out, failures included.

## Measured results

First run of the bank, on the code at commit `1f6ac5c`, with the sealed
defaults and seed `20261018`. Reproduce with
`REGIME_BANK=1 go test -run TestBank -v .` (about 30 seconds on a laptop).
Noise sigma is 1, the change is at sample 300, and a detection must come
within 50 samples. "Plain" replaces the empirical center of a new level with
the burn-in location.

| Case | Prior | False alarms / 1000 samples | Detected | Within bound | Median delay | Coverage | Median Since - tau |
|---|---|---|---|---|---|---|---|
| C1 no change | empirical | 0.437 | 0.0% | 0.0% | n/a | n/a | n/a |
| C1 no change | plain | 0.442 | 0.0% | 0.0% | n/a | n/a | n/a |
| C2 shift 2 sigma | empirical | 0.268 | 90.8% | 90.8% | 3.0 | 97.8% | 0.0 |
| C2 shift 2 sigma | plain | 0.270 | 90.9% | 90.9% | 3.0 | 97.7% | 0.0 |
| C3 shift 4 sigma | empirical | 0.212 | 100.0% | 100.0% | 0.0 | 99.2% | 0.0 |
| C3 shift 4 sigma | plain | 0.217 | 100.0% | 100.0% | 0.0 | 99.2% | 0.0 |
| C4 bend 0.1 sigma/sample | empirical | 0.298 | 53.9% | 53.6% | 16.0 | 93.1% | 12.0 |
| C4 bend 0.1 sigma/sample | plain | 0.300 | 53.9% | 53.6% | 16.0 | 93.5% | 12.0 |
| C6 shift 0.5 sigma | empirical | 0.277 | 7.3% | | 14.0 | 95.9% | 4.0 |

CUSUM (k = 1, true pre-change mean and sigma given), h = 3.406, C1 rate 0.437 / 1000 (regime 0.437)

| Shift | regime censored median delay | CUSUM censored median delay |
|---|---|---|
| 0.5 sigma | 51.0 | 51.0 |
| 2 sigma | 3.0 | 3.0 |
| 4 sigma | 0.0 | 1.0 |

### Sealed claims

1. C1 <= 1 false alarm / 1000: 0.437, PASS
2. C2 >= 95% detected and median delay <= 6.9: 90.8%, 3.0, FAIL
3. C3 >= 99% within 5: 100.0%, PASS
4. C4 >= 90% within 30: 53.6%, FAIL
5. coverage in [80%, 98%]: C2 97.8%, C3 99.2%, C4 93.1%, FAIL
6. CUSUM faster at 2 sigma (3.0 vs 3.0), regime faster at 0.5 (51.0 vs 51.0) and 4 sigma (0.0 vs 1.0): FAIL
7. empirical C1 rate >= plain: 0.437 vs 0.442, FAIL
8. C1 rates within a factor of 2: ratio 0.99, PASS
9. |median Since bias| <= 3, both priors: empirical 0.0/0.0/12.0, plain 0.0/0.0/12.0, FAIL
10. Original check: empirical prior moves coverage by < 10 points: max 0.4 points, PASS

### What the failures say

- **Claims 2 and 4 (missed detections).** About 9% of 2 sigma shifts and 46%
  of the slow bends are missed. In the missed shifts inspected, the `Trend`
  model explains the recent jump as a bend that started a little earlier: the
  posterior puts the start of the new regime 10 to 20 samples back, older than
  `Window`, so `ChangeProb` stays under the threshold even though the new level
  is already tracked. This is a property of measuring "started
  within the last `Window` samples", not a numerical error; a signal that
  compares against the regime of the last alarm is a candidate for the next
  version.
- **Claim 5.** The 90% interval is too wide for large jumps (99.2% coverage at
  4 sigma). It is calibrated at 2 sigma and for bends.
- **Claim 6.** At equal false-alarm rate, CUSUM, given the true pre-change
  mean and sigma, ties `regime` at 2 sigma; `regime` is faster at 4 sigma, and
  neither detects most 0.5 sigma shifts within 50 samples.
- **Claims 7 and 8.** The empirical center did not raise the false-alarm
  rate, so the predicted direction was wrong, but the effect is tiny: the two
  priors are within 3% of each other on every row.
- **Claim 9.** For bends, `Since` lands about 12 samples after the true start,
  because a bend with no jump becomes visible only as its effect accumulates.

## Benchmarks

`go test -bench .` on an Apple M5 Pro shared with other work, so the spread
is wide. `Observe` does not allocate.

| MaxRun | ns per Observe (3 runs) | allocs |
|---|---|---|
| 128 | 4,800 to 5,100 | 0 |
| 512 (default) | 19,600 to 55,000 | 0 |
| 2048 | 245,000 to 355,000 | 0 |

## Testing

Unit and property tests (posterior mass sums to 1, affine invariance, one
alarm per change, rejected samples leave the state unchanged, finite reports),
fuzzing of `New` and `Observe`, examples, `-race`, and mutation testing with
`python3 scripts/mutants.py`: 24 of 24 mutants killed (`MUTANTS.md`).

## Known limits

- Noise is assumed independent. Autocorrelated noise raises the false-alarm
  rate.
- Oscillation and seasonality are not a regime model. Remove the cycle first,
  or feed the residual of a forecaster.
- One large outlier can open a short regime. The debounce limits the damage.
- The hazard is per sample, not per unit of time.
- An alarm always comes some samples after the change. `Since` says when it
  probably started.
- State does not persist across process restarts. `Clone` covers what-if runs;
  persistence is planned for v2.

## License

Apache-2.0. See `LICENSE`.
