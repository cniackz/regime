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
were sealed before the first measurement. Results: see "Measured results"
below once the bank has run.

## Measured results

Pending the first bank run.

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
