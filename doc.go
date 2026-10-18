// Package regime detects changes of regime in a noisy series, online, one
// sample at a time.
//
// A regime is a stretch of the series that follows one level (model [Level])
// or one level plus one slope (model [Trend]), with Gaussian noise of unknown
// variance. At unknown times the level jumps or the slope bends. After every
// sample, [Detector.Observe] reports the probability that a new regime started
// recently, a single alarm per detected change, when the current regime
// probably started (with a credible interval), and the current level and
// slope.
//
//	d, err := regime.New()
//	if err != nil { ... }
//	for _, s := range samples {
//		r, err := d.Observe(s.T, s.V)
//		if err != nil { ... } // a rejected sample leaves d unchanged
//		if r.Alarm {
//			fmt.Printf("new regime since about %v (90%% interval %v to %v)\n",
//				r.Since, r.SinceLow, r.SinceHigh)
//		}
//	}
//
// # Method
//
// The detector is Bayesian online changepoint detection (Adams and MacKay
// 2007). Let r_t be the run length: the number of samples in the current
// regime, counting the sample at t. With a constant hazard H = 1/ExpectedRun
// (a geometric prior on regime length) the joint over run lengths evolves as
//
//	growth: P(r_t = r+1, x_1:t) = (1 - H) * pi_r(x_t) * P(r_{t-1} = r, x_1:t-1)
//	change: P(r_t = 1,   x_1:t) = H * pi_0(x_t) * sum_r P(r_{t-1} = r, x_1:t-1)
//
// and is normalized after each sample. Here pi_r is the posterior predictive
// density of x_t under the regime that has lasted r samples, and pi_0 is the
// prior predictive of a new regime. The change is placed just before x_t: x_t
// is scored under the prior and becomes the first sample of the new regime,
// and each growing regime absorbs x_t into its statistics (the convention of
// Fearnhead and Liu 2007; Adams and MacKay 2007 place the cut after x_t, which
// detects one sample later). Everything runs in log space with log-sum-exp.
//
// The reported [Report.ChangeProb] is P(r_t <= Window | x_1:t): the probability
// that the current regime started within the last Window samples. A single
// run length is a poor signal (in the Adams and MacKay form, the mass of a new
// run is exactly H at every step).
//
// # Regime model
//
// Within a regime, with local time u measured from the regime's first sample,
//
//	Trend: y = a + b*u + e,  e ~ N(0, sigma^2)
//	Level: y = a + e
//
// with a Normal-Inverse-Gamma prior on the coefficients and sigma^2. The prior
// is conjugate, so each run hypothesis keeps closed-form statistics (mean m,
// scaled covariance V, shape alpha, scale beta) and updates them with one
// rank-one step per sample, the recursive form of Bayesian linear regression
// (O'Hagan and Forster 2004; Bishop 2006, section 3.3; Murphy 2007):
//
//	s = 1 + x'Vx,  k = Vx/s,  e = y - x'm
//	m <- m + k*e,  V <- V - k(Vx)',  alpha <- alpha + 1/2,  beta <- beta + e^2/(2s)
//
// where x = (1, u) for Trend and x = (1) for Level. The posterior predictive of
// the next y is a Student-t with 2*alpha degrees of freedom, location x'm and
// squared scale (beta/alpha)*(1 + x'Vx).
//
// A new regime's prior centers its level on the predictive mean at that
// sample, weighted by the run-length posterior over all current hypotheses,
// with a wide spread. This is an empirical-Bayes choice: it assumes a new
// regime starts near where the series is, which keeps a pure bend (a slope
// change with no jump) detectable, and the wide spread keeps a jump cheap to
// explain. The weighted mean is smooth in the data, unlike the mean of the
// single most probable hypothesis, so it keeps the affine invariance below.
// Its effect on the false-alarm rate and on the calibration of
// [Report.SinceLow] and [Report.SinceHigh] is measured by the synthetic bank
// in the repository against a plain prior centered on the burn-in location,
// and the README reports both.
//
// The noise variance has an Inverse-Gamma prior with shape 2 and scale 1 in
// standardized units: prior mean 1, and a predictive with finite variance from
// a regime's first sample, so every reported spread is finite.
//
// # Units
//
// The first BurnIn samples are the burn-in. From them the detector fixes a
// location and scale for v (the median and the median absolute deviation,
// Hampel 1974; Rousseeuw and Croux 1993) and a time unit (the median sample
// spacing), and works on the standardized values from then on. Every field of
// a [Report] is converted back to the caller's units. As a result, an affine
// change of units in v or in t (positive scale factors) gives the same
// detections in real arithmetic. In float64, ChangeProb agrees to rounding,
// so Alarm and RunLength agree except when ChangeProb sits within rounding of
// the threshold or two hypotheses tie.
//
// # Burn-in and debounce
//
// BurnIn (default 30) is the burn-in length: no alarm and ChangeProb 0 before
// BurnIn samples. MinRun (default 10) is the debounce: after an alarm, the
// detector re-arms only once MinRun samples have passed and ChangeProb has
// fallen below the threshold, so two alarms are at least MinRun samples
// apart. A single change can still raise a second alarm later than that, when
// the posterior first explains it one way (for example as a bend) and then
// another; the README measures how often.
//
// # Cost
//
// The detector keeps at most MaxRun run hypotheses, oldest first, each with
// its own start time and statistics. After each sample it drops those whose
// posterior mass is below 1e-12 of the most probable one and, if still over
// MaxRun, the least probable one. Work per sample is O(MaxRun) and memory is
// fixed when the detector is built, so the hot path does not allocate. There
// is no cap on a hypothesis's length: a regime longer than MaxRun samples
// keeps refining its statistics, and its RunLength and Since stay exact.
//
// # What is not here
//
// There is no Merge: two detectors fed different stretches of a series have
// no meaningful union. There is no serialization in this version; [Detector.Clone]
// covers what-if runs in a process.
//
// # Limits
//
// Gradual bends are detected weakly in this version: on the synthetic bank
// in the repository, about half of the bends of 0.1 noise scales per sample
// are detected within 30 samples (the README has the measured limits).
// The noise is assumed independent; autocorrelated noise raises the
// false-alarm rate. Oscillation and seasonality are not a regime model:
// remove the cycle upstream. A single large outlier can open a short run; the
// debounce limits the damage. The hazard is per sample, not per unit of time.
// An online detector learns of a change after it happens: for a known level
// shift delta in noise sigma, no detector with an average run length to false
// alarm ARL0 has an average delay much below log(ARL0)/(delta^2/(2*sigma^2))
// (Lorden 1971; Lai 1998). Delays here count from the first sample of the
// new regime, which under the convention above is the sample scored by the
// prior. State does not persist across process restarts in this version:
// [Detector.Clone] covers what-if runs, and persistence is planned for v2.
//
// # References
//
//   - E. S. Page, "Continuous Inspection Schemes," Biometrika, 1954.
//   - G. Lorden, "Procedures for Reacting to a Change in Distribution," Annals
//     of Mathematical Statistics, 1971.
//   - F. R. Hampel, "The Influence Curve and Its Role in Robust Estimation,"
//     JASA, 1974.
//   - P. J. Rousseeuw, C. Croux, "Alternatives to the Median Absolute
//     Deviation," JASA, 1993.
//   - T. L. Lai, "Information Bounds and Quick Detection of Parameter Changes in
//     Stochastic Systems," IEEE Transactions on Information Theory, 1998.
//   - A. O'Hagan, J. Forster, "Kendall's Advanced Theory of Statistics, Vol. 2B:
//     Bayesian Inference," 2nd ed., 2004.
//   - C. M. Bishop, "Pattern Recognition and Machine Learning," 2006.
//   - K. P. Murphy, "Conjugate Bayesian Analysis of the Gaussian Distribution,"
//     technical note, 2007.
//   - R. P. Adams, D. J. C. MacKay, "Bayesian Online Changepoint Detection,"
//     arXiv:0710.3742, 2007.
//   - P. Fearnhead, Z. Liu, "On-line Inference for Multiple Changepoint
//     Problems," JRSS B, 2007.
//   - R. Killick, P. Fearnhead, I. A. Eckley, "Optimal Detection of
//     Changepoints With a Linear Computational Cost," JASA, 2012.
package regime
