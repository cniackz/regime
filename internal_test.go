package regime

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// The recursive update must match the batch Normal-Inverse-Gamma posterior
// (O'Hagan and Forster 2004): for the Level model with prior (m0, V0, a0, b0)
// and data y_1..y_n, Vn = 1/(1/V0 + n), mn = Vn (m0/V0 + sum y),
// an = a0 + n/2, bn = b0 + (sum y^2 + m0^2/V0 - mn^2/Vn)/2.
func TestUpdateMatchesBatchPosterior(t *testing.T) {
	ys := []float64{1, 2.5, -0.5, 3, 0.25}
	m0 := 0.7
	r := newRun(Level, 0, 0, m0)
	var sy, syy float64
	for _, y := range ys {
		r.update([2]float64{1, 0}, y)
		sy += y
		syy += y * y
	}
	n := float64(len(ys))
	vn := 1 / (1/priorLevelVar + n)
	mn := vn * (m0/priorLevelVar + sy)
	an := priorAlpha + n/2
	bn := priorBeta + (syy+m0*m0/priorLevelVar-mn*mn/vn)/2
	near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-12*(1+math.Abs(b)) }
	if !near(r.v[0][0], vn) || !near(r.m[0], mn) || !near(r.alpha, an) || !near(r.beta, bn) {
		t.Fatalf("got V=%v m=%v a=%v b=%v, want %v %v %v %v", r.v[0][0], r.m[0], r.alpha, r.beta, vn, mn, an, bn)
	}
	loc, s2, nu := r.predict([2]float64{1, 0})
	if !near(loc, mn) || !near(s2, bn/an*(1+vn)) || !near(nu, 2*an) {
		t.Fatalf("predictive %v %v %v", loc, s2, nu)
	}
}

func TestPruneDropsLeastProbable(t *testing.T) {
	rs := []run{{startT: 1, logp: math.Log(0.5)}, {startT: 2, logp: math.Log(0.1)}, {startT: 3, logp: math.Log(0.4)}}
	rs = prune(rs, 2)
	if len(rs) != 2 || rs[0].startT != 1 || rs[1].startT != 3 {
		t.Fatalf("kept %+v", rs)
	}
	if math.Abs(rs[0].p+rs[1].p-1) > 1e-12 {
		t.Fatal("not renormalized")
	}
}

func TestChangeProbWindowInclusive(t *testing.T) {
	d, _ := New(WithWindow(3))
	d.runs = []run{
		{n: 3, p: 0.25}, {n: 4, p: 0.5}, {n: 1, p: 0.125}, {n: 2, p: 0.125, initial: true},
	}
	if got := d.changeProb(); got != 0.375 {
		t.Fatalf("changeProb = %v, want 0.375 (n <= Window counts, initial does not)", got)
	}
}

func TestNoAlarmRightAfterShortBurnIn(t *testing.T) {
	// With BurnIn below Window, the initial regime is shorter than Window
	// right after burn-in; it is not a change and must not count.
	d, _ := New(WithBurnIn(5), WithWindow(10))
	vs := noisy(11, 40, func(int) float64 { return 0 })
	for i, v := range vs {
		r, _ := d.Observe(float64(i), v)
		if r.Alarm {
			t.Fatalf("alarm at %d", i)
		}
	}
}

func TestThresholdIsInclusive(t *testing.T) {
	vs := noisy(12, 200, func(i int) float64 {
		if i >= 100 {
			return 1.5
		}
		return 0
	})
	d, _ := New(WithThreshold(1))
	best, at := 0.0, -1
	for i, v := range vs {
		r, _ := d.Observe(float64(i), v)
		if r.ChangeProb > best {
			best, at = r.ChangeProb, i
		}
	}
	d2, _ := New(WithThreshold(best))
	for i, v := range vs {
		r, _ := d2.Observe(float64(i), v)
		if r.Alarm != (i == at) {
			t.Fatalf("threshold %v: alarm=%v at %d, want only at %d", best, r.Alarm, i, at)
		}
	}
}

func TestDebounceHoldsUntilMinRun(t *testing.T) {
	// Window 1 lets ChangeProb fall right after the first step, so only the
	// debounce stops a second alarm on the step four samples later.
	d, _ := New(WithWindow(1), WithMinRun(10))
	var alarms []int
	for i := 0; i < 200; i++ {
		v := 0.0
		switch {
		case i >= 104:
			v = 1000
		case i >= 100:
			v = 5
		}
		r, _ := d.Observe(float64(i), v)
		if r.Alarm {
			alarms = append(alarms, i)
		}
	}
	if !reflect.DeepEqual(alarms, []int{100}) {
		t.Fatalf("alarms %v, want [100]", alarms)
	}
}

func TestSinceIntervalHoldsNinetyPercent(t *testing.T) {
	vs := noisy(13, 500, func(i int) float64 {
		if i >= 250 {
			return 1
		}
		return 0
	})
	d, _ := New()
	for i, v := range vs {
		r, _ := d.Observe(float64(i), v)
		if len(d.runs) < 2 {
			continue
		}
		var in float64
		for _, h := range d.runs {
			if h.startT >= r.SinceLow && h.startT <= r.SinceHigh {
				in += h.p
			}
		}
		if in < 0.90-1e-9 {
			t.Fatalf("sample %d: interval [%v, %v] holds %v of the mass", i, r.SinceLow, r.SinceHigh, in)
		}
		if r.SinceLow > r.Since || r.Since > r.SinceHigh {
			t.Logf("sample %d: Since %v outside [%v, %v] (the MAP can sit in a tail)", i, r.Since, r.SinceLow, r.SinceHigh)
		}
	}
}

func TestNonFiniteStateIsRejected(t *testing.T) {
	d, _ := New(WithBurnIn(3))
	for i := range 3 {
		d.Observe(float64(i), float64(i))
	}
	d.runs[0].beta = math.Inf(1) // a poisoned hypothesis; Inf, since DeepEqual never matches NaN
	before := d.Clone()
	if _, err := d.Observe(10, 1); !errors.Is(err, ErrRange) {
		t.Fatalf("err = %v, want ErrRange", err)
	}
	if !reflect.DeepEqual(d, before) {
		t.Fatal("rejected step changed state")
	}
}

func TestSinceQuantiles(t *testing.T) {
	d, _ := New(WithBurnIn(3))
	for i := range 3 {
		d.Observe(float64(i), float64(i%2))
	}
	base := d.runs[0]
	d.runs = d.runs[:0]
	for k, p := range []float64{0.02, 0.28, 0.3, 0.38, 0.02} {
		r := base
		r.startT, r.p, r.logp = float64(10*k), p, math.Log(p)
		d.runs = append(d.runs, r)
	}
	rep := d.report(0, false)
	if rep.SinceLow != 10 || rep.SinceHigh != 30 || rep.Since != 30 {
		t.Fatalf("since %v [%v, %v], want 30 [10, 30]", rep.Since, rep.SinceLow, rep.SinceHigh)
	}
}
