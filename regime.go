package regime

import (
	"errors"
	"math"
	"slices"
)

// Model selects what a regime looks like.
type Model int

const (
	// Trend is piecewise linear: a regime is one level and one slope plus
	// noise. It is the zero value and the default.
	Trend Model = iota
	// Level is a piecewise constant mean: a regime is one level plus noise.
	Level
)

// String returns "Trend" or "Level".
func (m Model) String() string {
	switch m {
	case Trend:
		return "Trend"
	case Level:
		return "Level"
	}
	return "Model(invalid)"
}

// Defaults used by [New] with no options and by the zero [Detector].
const (
	DefaultModel       = Trend
	DefaultExpectedRun = 250.0
	DefaultMaxRun      = 512
	DefaultThreshold   = 0.5
	DefaultWindow      = 10
	DefaultBurnIn      = 30
	DefaultMinRun      = 10
)

// Limits on option values.
const (
	maxMaxRun      = 1 << 16
	maxBurnIn      = 1 << 16
	maxMinRun      = 1 << 16
	maxWindow      = 1 << 20
	maxExpectedRun = 1e12
)

// Prior of a new regime, in standardized units (see the package doc).
const (
	priorLevelVar = 100.0 // spread of the new level: 10 burn-in noise scales
	priorSlopeVar = 1.0   // spread of the new slope: 1 noise scale per sample spacing
	priorAlpha    = 2.0   // Inverse-Gamma shape of the noise variance; > 1 keeps spreads finite
	priorBeta     = 1.0   // Inverse-Gamma scale of the noise variance; prior mean 1
)

// pruneLog is log(1e-12): hypotheses below this mass relative to the most
// probable one are dropped.
var pruneLog = math.Log(1e-12)

// maxStd bounds the magnitude of standardized values and times. Beyond it,
// squares in the update could overflow, so the sample is rejected.
const maxStd = 1e100

// Errors returned by [New] and [Detector.Observe].
var (
	ErrNonFinite = errors.New("regime: t or v is NaN or infinite")
	ErrTimeOrder = errors.New("regime: t is not greater than the previous t")
	ErrRange     = errors.New("regime: sample out of range after standardization")
	ErrOption    = errors.New("regime: invalid option")
)

type config struct {
	set         bool
	model       Model
	expectedRun float64
	maxRun      int
	threshold   float64
	window      int
	burnIn      int
	minRun      int
	plainPrior  bool // bank hook: center new levels on the burn-in location
}

func defaultConfig() config {
	return config{
		set:         true,
		model:       DefaultModel,
		expectedRun: DefaultExpectedRun,
		maxRun:      DefaultMaxRun,
		threshold:   DefaultThreshold,
		window:      DefaultWindow,
		burnIn:      DefaultBurnIn,
		minRun:      DefaultMinRun,
	}
}

func (c config) valid() bool {
	return (c.model == Trend || c.model == Level) &&
		c.expectedRun > 1 && c.expectedRun <= maxExpectedRun &&
		c.maxRun >= 2 && c.maxRun <= maxMaxRun &&
		c.threshold > 0 && c.threshold <= 1 &&
		c.window >= 1 && c.window <= maxWindow &&
		c.burnIn >= 2 && c.burnIn <= maxBurnIn &&
		c.minRun >= 1 && c.minRun <= maxMinRun
}

// Option configures a [Detector] built by [New].
type Option func(*config)

// WithModel sets the regime model. Default [Trend].
func WithModel(m Model) Option { return func(c *config) { c.model = m } }

// WithExpectedRun sets the prior mean regime length, in samples: the hazard is
// 1/n per sample. A larger n means fewer false alarms and a longer delay.
// It must be in (1, 1e12]. Default 250.
func WithExpectedRun(n float64) Option { return func(c *config) { c.expectedRun = n } }

// WithMaxRun sets how many run-length hypotheses are kept, which bounds the
// work per sample. It must be in [2, 65536]. Default 512.
func WithMaxRun(n int) Option { return func(c *config) { c.maxRun = n } }

// WithThreshold sets the alarm level for [Report.ChangeProb]. It must be in
// (0, 1]. Default 0.5.
func WithThreshold(p float64) Option { return func(c *config) { c.threshold = p } }

// WithWindow sets how recent, in samples, a regime start must be to count in
// [Report.ChangeProb]. It must be in [1, 1<<20]. Default 10.
func WithWindow(w int) Option { return func(c *config) { c.window = w } }

// WithBurnIn sets the burn-in length, in samples: the samples that fix the
// units, with no alarm and ChangeProb 0. It must be in [2, 65536]. Default 30.
func WithBurnIn(n int) Option { return func(c *config) { c.burnIn = n } }

// WithMinRun sets the debounce, in samples: two alarms are always at least
// this many samples apart. It must be in [1, 65536]. Default 10.
func WithMinRun(m int) Option { return func(c *config) { c.minRun = m } }

// run is one run-length hypothesis: a regime that started at startT.
type run struct {
	startT  float64    // caller time of the first sample
	startU  float64    // standardized time of the first sample
	n       int        // samples in the regime, counting the latest
	logp    float64    // log posterior mass
	p       float64    // posterior mass, exp(logp), set by normalize
	m       [2]float64 // coefficient mean: level at u=0, slope
	v       [2][2]float64
	alpha   float64
	beta    float64
	initial bool // the regime the series started in; not a change
}

func newRun(model Model, startT, startU, center float64) run {
	r := run{startT: startT, startU: startU, m: [2]float64{center, 0}, alpha: priorAlpha, beta: priorBeta}
	r.v[0][0] = priorLevelVar
	if model == Trend {
		r.v[1][1] = priorSlopeVar
	}
	return r
}

// predict returns the location, squared scale and degrees of freedom of the
// Student-t posterior predictive at regressor x (Murphy 2007; Bishop 2006,
// section 3.3): location x'm, squared scale (beta/alpha)(1 + x'Vx), 2*alpha
// degrees of freedom.
func (r *run) predict(x [2]float64) (loc, scale2, nu float64) {
	vx0 := r.v[0][0]*x[0] + r.v[0][1]*x[1]
	vx1 := r.v[1][0]*x[0] + r.v[1][1]*x[1]
	// x'Vx >= 0 in exact arithmetic; clamp the rounding of a long regime.
	q := max(x[0]*vx0+x[1]*vx1, 0)
	loc = x[0]*r.m[0] + x[1]*r.m[1]
	scale2 = r.beta / r.alpha * (1 + q)
	nu = 2 * r.alpha
	return loc, scale2, nu
}

// update folds sample y at regressor x into the Normal-Inverse-Gamma
// statistics with the recursive Bayesian least-squares step (Bishop 2006,
// section 3.3; O'Hagan and Forster 2004):
//
//	s = 1 + x'Vx, k = Vx/s, e = y - x'm
//	m += k e, V -= k (Vx)', alpha += 1/2, beta += e^2/(2s)
func (r *run) update(x [2]float64, y float64) {
	vx0 := r.v[0][0]*x[0] + r.v[0][1]*x[1]
	vx1 := r.v[1][0]*x[0] + r.v[1][1]*x[1]
	s := 1 + max(x[0]*vx0+x[1]*vx1, 0)
	e := y - (x[0]*r.m[0] + x[1]*r.m[1])
	k0, k1 := vx0/s, vx1/s
	r.m[0] += k0 * e
	r.m[1] += k1 * e
	r.v[0][0] -= k0 * vx0
	r.v[0][1] -= k0 * vx1
	r.v[1][0] = r.v[0][1]
	r.v[1][1] -= k1 * vx1
	r.v[0][0] = max(r.v[0][0], 0)
	r.v[1][1] = max(r.v[1][1], 0)
	r.alpha += 0.5
	r.beta += e * e / (2 * s)
	r.n++
}

func allFinite(rs []run) bool {
	for i := range rs {
		if !rs[i].finite() {
			return false
		}
	}
	return true
}

func (r *run) finite() bool {
	return isFinite(r.logp) && isFinite(r.p) && isFinite(r.m[0]) && isFinite(r.m[1]) &&
		isFinite(r.v[0][0]) && isFinite(r.v[0][1]) && isFinite(r.v[1][1]) &&
		isFinite(r.beta)
}

// logStudentT is the log density of a Student-t with nu degrees of freedom,
// location loc and squared scale scale2, at y.
func logStudentT(y, loc, scale2, nu float64) float64 {
	return logStudentTNorm(lgammaRatio(nu), y, loc, scale2, nu)
}

// lgammaRatio is log Gamma((nu+1)/2) - log Gamma(nu/2), the normalizing term
// of the Student-t density.
func lgammaRatio(nu float64) float64 {
	a, _ := math.Lgamma((nu + 1) / 2)
	b, _ := math.Lgamma(nu / 2)
	return a - b
}

// logStudentTNorm is logStudentT with its lgammaRatio term given.
func logStudentTNorm(c, y, loc, scale2, nu float64) float64 {
	z := y - loc
	return c - 0.5*math.Log(nu*math.Pi*scale2) - (nu+1)/2*math.Log1p(z*z/(nu*scale2))
}

func isFinite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

// Report is what [Detector.Observe] returns after each accepted sample. All
// times, levels and slopes are in the caller's units. Every field is finite.
type Report struct {
	// N is the number of samples accepted so far, counting this one.
	N int
	// ChangeProb is the posterior probability that the current regime started
	// within the last Window samples. It is 0 during burn-in.
	ChangeProb float64
	// Alarm is true on the sample where a change is detected. Two alarms are
	// at least MinRun samples apart, and there is none during burn-in.
	Alarm bool
	// RunLength is the length, in samples, of the most probable current
	// regime.
	RunLength int
	// Since is the start time of the most probable current regime.
	Since float64
	// SinceLow and SinceHigh are the 5% and 95% posterior quantiles of the
	// current regime's start time.
	SinceLow, SinceHigh float64
	// Level is the posterior mean of the most probable regime at this sample.
	Level float64
	// Slope is the posterior mean slope of the most probable regime, per unit
	// of t. It is 0 for the Level model.
	Slope float64
	// SlopeStd is the posterior standard deviation of Slope. It is 0 for the
	// Level model.
	SlopeStd float64
}

// Detector is an online changepoint detector. It is not safe for concurrent
// use. The zero value is ready to use and takes every default on its first
// [Detector.Observe]; [New] sets options.
type Detector struct {
	cfg   config
	ready bool

	n     int     // samples accepted
	lastT float64 // time of the last accepted sample

	// Burn-in buffers, then the standardization they fix.
	burnT, burnV, scratch []float64
	m0, s0, t0, dt0       float64

	runs, next []run // run hypotheses, oldest first; next is scratch, len 0 when idle

	armed        bool
	lastAlarm    int // N of the last alarm, for the debounce
	logH, log1mH float64

	// lgc[n] caches lgammaRatio for a run that has absorbed n samples, whose
	// predictive has 2*priorAlpha + n degrees of freedom.
	lgc []float64
}

// New returns a Detector with the given options applied over the defaults. An
// invalid option value returns [ErrOption].
func New(opts ...Option) (*Detector, error) {
	c := defaultConfig()
	for _, o := range opts {
		if o != nil {
			o(&c)
		}
	}
	if !c.valid() {
		return nil, ErrOption
	}
	d := &Detector{cfg: c}
	d.build()
	return d, nil
}

func (d *Detector) build() {
	if !d.cfg.set {
		d.cfg = defaultConfig()
	}
	c := d.cfg
	d.burnT = make([]float64, 0, c.burnIn)
	d.burnV = make([]float64, 0, c.burnIn)
	d.scratch = make([]float64, 0, c.burnIn)
	d.runs = make([]run, 0, c.maxRun+1)
	d.next = make([]run, 0, c.maxRun+1)
	h := 1 / c.expectedRun
	d.logH = math.Log(h)
	d.log1mH = math.Log1p(-h)
	d.lgc = make([]float64, 2*c.maxRun+c.burnIn+2)
	for n := range d.lgc {
		d.lgc[n] = lgammaRatio(2*priorAlpha + float64(n))
	}
	d.armed = true
	d.ready = true
}

func (d *Detector) lgammaRatio(n int, nu float64) float64 {
	if n < len(d.lgc) {
		return d.lgc[n]
	}
	return lgammaRatio(nu)
}

// Reset returns the detector to its state right after construction, keeping
// its options and buffers.
func (d *Detector) Reset() {
	if !d.ready {
		return
	}
	d.n = 0
	d.lastT = 0
	d.burnT = d.burnT[:0]
	d.burnV = d.burnV[:0]
	d.scratch = d.scratch[:0]
	d.m0, d.s0, d.t0, d.dt0 = 0, 0, 0, 0
	d.runs = d.runs[:0]
	d.next = d.next[:0]
	d.armed = true
	d.lastAlarm = 0
}

// Clone returns an independent copy of the detector.
func (d *Detector) Clone() *Detector {
	c := *d
	if d.ready {
		c.burnT = append(make([]float64, 0, cap(d.burnT)), d.burnT...)
		c.burnV = append(make([]float64, 0, cap(d.burnV)), d.burnV...)
		c.scratch = make([]float64, 0, cap(d.scratch))
		c.runs = append(make([]run, 0, cap(d.runs)), d.runs...)
		c.next = make([]run, 0, cap(d.next))
	}
	return &c
}

// Observe adds the sample (t, v) and reports the detector's state after it.
// The time t must be greater than the previous accepted t. On error the
// detector is left as it was before the call.
func (d *Detector) Observe(t, v float64) (Report, error) {
	if !isFinite(t) || !isFinite(v) {
		return Report{}, ErrNonFinite
	}
	if d.n > 0 && !(t > d.lastT) {
		return Report{}, ErrTimeOrder
	}
	if !d.ready {
		d.build()
	}
	if len(d.runs) == 0 {
		return d.burnIn(t, v)
	}
	return d.step(t, v)
}

// burnIn stores a burn-in sample and, on the last one, fixes the
// standardization and the initial regime.
func (d *Detector) burnIn(t, v float64) (Report, error) {
	d.burnT = append(d.burnT, t)
	d.burnV = append(d.burnV, v)
	if len(d.burnT) < d.cfg.burnIn {
		d.n++
		d.lastT = t
		return Report{
			N: d.n, RunLength: d.n, Level: v,
			Since: d.burnT[0], SinceLow: d.burnT[0], SinceHigh: d.burnT[0],
		}, nil
	}
	undo := func() (Report, error) {
		d.burnT = d.burnT[:len(d.burnT)-1]
		d.burnV = d.burnV[:len(d.burnV)-1]
		d.m0, d.s0, d.t0, d.dt0 = 0, 0, 0, 0
		return Report{}, ErrRange
	}
	if !d.standardize() {
		return undo()
	}
	r := newRun(d.cfg.model, d.burnT[0], 0, 0)
	r.initial = true
	r.p = 1
	for i := range d.burnT {
		y := (d.burnV[i] - d.m0) / d.s0
		u := (d.burnT[i] - d.t0) / d.dt0
		r.update(d.regressor(u), y)
	}
	if !r.finite() {
		return undo()
	}
	d.runs = append(d.runs, r)
	d.n++
	d.lastT = t
	return d.report(0, false), nil
}

// standardize fixes location, scale and time unit from the burn-in buffers.
// It reports false if a standardized burn-in value or time is out of range.
func (d *Detector) standardize() bool {
	s := append(d.scratch[:0], d.burnV...)
	slices.Sort(s)
	med := median(s)
	for i := range s {
		s[i] = math.Abs(s[i] - med)
	}
	slices.Sort(s)
	scale := 1.4826 * median(s) // MAD, consistent for the Gaussian (Hampel 1974)
	if !(scale > 0) || !isFinite(scale) {
		var sum float64
		for _, x := range s {
			sum += x / float64(len(s))
		}
		scale = math.Sqrt(math.Pi/2) * sum // mean absolute deviation
	}
	if !(scale > 0) || !isFinite(scale) {
		scale = 1
	}
	s = s[:0]
	for i := 1; i < len(d.burnT); i++ {
		s = append(s, d.burnT[i]-d.burnT[i-1])
	}
	slices.Sort(s)
	dt := median(s)
	d.scratch = s[:0]
	if !(dt > 0) || !isFinite(dt) {
		return false
	}
	d.m0, d.s0, d.t0, d.dt0 = med, scale, d.burnT[0], dt
	for i := range d.burnT {
		y := (d.burnV[i] - d.m0) / d.s0
		u := (d.burnT[i] - d.t0) / d.dt0
		if !inRange(y) || !inRange(u) {
			return false
		}
	}
	return true
}

func median(sorted []float64) float64 {
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return sorted[n/2-1]/2 + sorted[n/2]/2
}

func inRange(f float64) bool { return isFinite(f) && math.Abs(f) <= maxStd }

func (d *Detector) regressor(u float64) [2]float64 {
	if d.cfg.model == Trend {
		return [2]float64{1, u}
	}
	return [2]float64{1, 0}
}

// step runs one changepoint update: Adams and MacKay 2007 with the change
// placed before the sample (Fearnhead and Liu 2007), see the package doc.
func (d *Detector) step(t, v float64) (Report, error) {
	y := (v - d.m0) / d.s0
	ut := (t - d.t0) / d.dt0
	if !inRange(y) || !inRange(ut) {
		return Report{}, ErrRange
	}

	next := d.next[:0]
	var center float64 // posterior-weighted predictive mean (empirical Bayes)
	for i := range d.runs {
		r := d.runs[i]
		x := d.regressor(ut - r.startU)
		loc, s2, nu := r.predict(x)
		center += r.p * loc
		r.logp += logStudentTNorm(d.lgammaRatio(r.n, nu), y, loc, s2, nu) + d.log1mH
		r.update(x, y)
		next = append(next, r)
	}
	if d.cfg.plainPrior {
		center = 0
	}
	nr := newRun(d.cfg.model, t, ut, center)
	x0 := [2]float64{1, 0}
	loc, s2, nu := nr.predict(x0)
	// The previous posterior sums to 1, so the change term is H * pi_0(y).
	nr.logp = d.logH + logStudentTNorm(d.lgc[0], y, loc, s2, nu)
	nr.update(x0, y)
	next = append(next, nr)

	// Check before pruning too: a mass of -Inf or NaN would make prune drop
	// every hypothesis.
	if !allFinite(next) {
		d.next = next[:0]
		return Report{}, ErrRange
	}
	next = prune(next, d.cfg.maxRun)
	if len(next) == 0 || !allFinite(next) {
		d.next = next[:0]
		return Report{}, ErrRange
	}

	d.runs, d.next = next, d.runs[:0]
	d.n++
	d.lastT = t

	cp := d.changeProb()
	alarm := false
	// Debounce: after an alarm, re-arm only once MinRun samples have passed
	// since it and ChangeProb has fallen below the threshold. Counting from
	// the alarm, not from the most probable regime, matters: right after an
	// alarm the most probable hypothesis can flip back to an old, long regime.
	if d.armed && cp >= d.cfg.threshold {
		alarm = true
		d.armed = false
		d.lastAlarm = d.n
	} else if !d.armed && cp < d.cfg.threshold && d.n-d.lastAlarm >= d.cfg.minRun {
		d.armed = true
	}
	return d.report(cp, alarm), nil
}

// normalize rescales log masses so they sum to 1 (log-sum-exp) and sets the
// linear masses.
func normalize(rs []run) {
	mx := math.Inf(-1)
	for i := range rs {
		mx = max(mx, rs[i].logp)
	}
	var sum float64
	for i := range rs {
		rs[i].p = math.Exp(rs[i].logp - mx)
		sum += rs[i].p
	}
	lse := mx + math.Log(sum)
	for i := range rs {
		rs[i].logp -= lse
		rs[i].p /= sum
	}
}

// prune drops hypotheses below 1e-12 of the most probable one, then the least
// probable ones until at most maxRun remain, and renormalizes. Order is kept.
func prune(rs []run, maxRun int) []run {
	mx := math.Inf(-1)
	for i := range rs {
		mx = max(mx, rs[i].logp)
	}
	k := 0
	for i := range rs {
		if rs[i].logp-mx >= pruneLog {
			rs[k] = rs[i]
			k++
		}
	}
	rs = rs[:k]
	for len(rs) > maxRun {
		lo := 0
		for i := range rs {
			if rs[i].logp < rs[lo].logp {
				lo = i
			}
		}
		rs = slices.Delete(rs, lo, lo+1)
	}
	normalize(rs)
	return rs
}

func (d *Detector) mapIndex() int {
	best := 0
	for i := range d.runs {
		if d.runs[i].logp > d.runs[best].logp {
			best = i
		}
	}
	return best
}

func (d *Detector) changeProb() float64 {
	var p float64
	for i := range d.runs {
		r := &d.runs[i]
		if !r.initial && r.n <= d.cfg.window {
			p += r.p
		}
	}
	return min(p, 1)
}

func (d *Detector) report(cp float64, alarm bool) Report {
	r := &d.runs[d.mapIndex()]
	ut := (d.lastT - d.t0) / d.dt0
	lvl, _, _ := r.predict(d.regressor(ut - r.startU))

	// Quantiles of the start time: hypotheses are oldest first, so the
	// cumulative mass runs from the earliest start to the latest.
	low, high := d.runs[0].startT, d.runs[len(d.runs)-1].startT
	var cum float64
	lowSet := false
	for i := range d.runs {
		cum += d.runs[i].p
		if !lowSet && cum >= 0.05 {
			low, lowSet = d.runs[i].startT, true
		}
		if cum >= 0.95 {
			high = d.runs[i].startT
			break
		}
	}

	rep := Report{
		N:          d.n,
		ChangeProb: cp,
		Alarm:      alarm,
		RunLength:  r.n,
		Since:      r.startT,
		SinceLow:   low,
		SinceHigh:  high,
		Level:      d.m0 + d.s0*lvl,
	}
	if d.cfg.model == Trend {
		rep.Slope = d.s0 / d.dt0 * r.m[1]
		rep.SlopeStd = d.s0 / d.dt0 * math.Sqrt(max(r.beta/(r.alpha-1)*r.v[1][1], 0))
	}
	if !isFinite(rep.Level) || !isFinite(rep.Slope) || !isFinite(rep.SlopeStd) {
		// Not expected for samples in range; kept so a Report is always finite.
		rep.Level, rep.Slope, rep.SlopeStd = 0, 0, 0
	}
	return rep
}
