package regime

import (
	"math"
	"reflect"
	"testing"
)

func finiteReport(r Report) bool {
	for _, f := range []float64{r.ChangeProb, r.Since, r.SinceLow, r.SinceHigh, r.Level, r.Slope, r.SlopeStd} {
		if !isFinite(f) {
			return false
		}
	}
	return r.ChangeProb >= 0 && r.ChangeProb <= 1 && r.SinceLow <= r.SinceHigh
}

func TestPosteriorSumsToOne(t *testing.T) {
	for _, m := range []Model{Trend, Level} {
		d, _ := New(WithModel(m))
		vs := noisy(5, 800, func(i int) float64 { return float64(i/200) * 2 })
		for i, v := range vs {
			r, _ := d.Observe(float64(i), v)
			if !finiteReport(r) {
				t.Fatalf("%v sample %d: non-finite report %+v", m, i, r)
			}
			if len(d.runs) == 0 {
				continue
			}
			var s float64
			for _, h := range d.runs {
				if h.logp > 0 {
					t.Fatalf("log mass %v > 0", h.logp)
				}
				s += math.Exp(h.logp)
			}
			if math.Abs(s-1) > 1e-9 {
				t.Fatalf("%v sample %d: mass sums to %v", m, i, s)
			}
			if len(d.runs) > d.cfg.maxRun {
				t.Fatalf("%d hypotheses > MaxRun", len(d.runs))
			}
			for j := 1; j < len(d.runs); j++ {
				if d.runs[j].startT <= d.runs[j-1].startT {
					t.Fatal("hypotheses not oldest first")
				}
			}
		}
	}
}

func TestNoiseFreeStepAlarmsOnStep(t *testing.T) {
	for _, m := range []Model{Trend, Level} {
		d, _ := New(WithModel(m))
		for i := 0; i < 200; i++ {
			v := 0.0
			if i >= 100 {
				v = 5
			}
			r, err := d.Observe(float64(i), v)
			if err != nil {
				t.Fatal(err)
			}
			if r.Alarm != (i == 100) {
				t.Fatalf("%v: sample %d alarm=%v (cp %v)", m, i, r.Alarm, r.ChangeProb)
			}
		}
	}
}

func TestNoiseFreeBendAlarmsQuickly(t *testing.T) {
	d, _ := New()
	at := -1
	for i := 0; i < 300; i++ {
		v := 0.0
		if i >= 100 {
			v = 0.1 * float64(i-100)
		}
		// A tiny deterministic ripple keeps the burn-in scale above zero.
		v += 1e-3 * math.Sin(float64(i))
		r, _ := d.Observe(float64(i), v)
		if r.Alarm {
			if at >= 0 {
				t.Fatalf("second alarm at %d", i)
			}
			at = i
		}
	}
	if at < 100 || at > 110 {
		t.Fatalf("bend alarm at %d, want in [100, 110]", at)
	}
}

func TestConstantNeverAlarms(t *testing.T) {
	for _, c := range []float64{0, 7, -1e9} {
		d, _ := New()
		for i := 0; i < 2000; i++ {
			r, _ := d.Observe(float64(i), c)
			if r.Alarm || !finiteReport(r) {
				t.Fatalf("c=%v sample %d: %+v", c, i, r)
			}
			if i >= 30 && math.Abs(r.Level-c) > 1e-9*math.Max(1, math.Abs(c)) {
				t.Fatalf("level %v, want %v", r.Level, c)
			}
		}
	}
}

func TestOneAlarmPerChange(t *testing.T) {
	vs := noisy(6, 1200, func(i int) float64 { return float64(i/300) * 6 })
	d, _ := New()
	rs := feed(t, d, vs)
	var alarms []int
	for i, r := range rs {
		if r.Alarm {
			alarms = append(alarms, i)
		}
	}
	if len(alarms) != 3 {
		t.Fatalf("alarms at %v, want one per change at 300, 600, 900", alarms)
	}
	for k, a := range alarms {
		if a < 300*(k+1) || a > 300*(k+1)+5 {
			t.Fatalf("alarm %d at %d", k, a)
		}
		if math.Abs(rs[a].Since-float64(300*(k+1))) > 3 {
			t.Fatalf("Since %v for change at %d", rs[a].Since, 300*(k+1))
		}
	}
}

// Affine invariance (package doc, "Units"): with positive scale factors the
// detections match, ChangeProb to rounding, and levels and times transform.
func TestAffineInvariance(t *testing.T) {
	const c, off, e, f = 3.5, -1234.5, 0.25, 1e4
	vs := noisy(8, 900, func(i int) float64 {
		switch {
		case i >= 600:
			return 4 - 0.02*float64(i-600)
		case i >= 300:
			return 3
		}
		return 0
	})
	a, _ := New()
	b, _ := New()
	for i, v := range vs {
		ra, _ := a.Observe(float64(i), v)
		rb, err := b.Observe(e*float64(i)+f, c*v+off)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(ra.ChangeProb-rb.ChangeProb) > 1e-9 {
			t.Fatalf("sample %d: ChangeProb %v vs %v", i, ra.ChangeProb, rb.ChangeProb)
		}
		margin := math.Abs(ra.ChangeProb - DefaultThreshold)
		if margin > 1e-6 && ra.Alarm != rb.Alarm {
			t.Fatalf("sample %d: alarm %v vs %v", i, ra.Alarm, rb.Alarm)
		}
		if ra.RunLength != rb.RunLength {
			t.Logf("sample %d: run length %d vs %d (tie)", i, ra.RunLength, rb.RunLength)
			continue
		}
		tol := 1e-6 * (1 + math.Abs(c*ra.Level+off))
		if math.Abs(c*ra.Level+off-rb.Level) > tol {
			t.Fatalf("sample %d: level %v vs %v", i, c*ra.Level+off, rb.Level)
		}
		if math.Abs(e*ra.Since+f-rb.Since) > 1e-6*(1+math.Abs(rb.Since)) {
			t.Fatalf("sample %d: since %v vs %v", i, e*ra.Since+f, rb.Since)
		}
		if math.Abs(c/e*ra.Slope-rb.Slope) > 1e-6*(1+math.Abs(rb.Slope)) {
			t.Fatalf("sample %d: slope %v vs %v", i, c/e*ra.Slope, rb.Slope)
		}
	}
}

func TestRejectedSampleLeavesState(t *testing.T) {
	d, _ := New(WithBurnIn(5))
	vs := noisy(9, 60, func(int) float64 { return 0 })
	bad := [][2]float64{{math.NaN(), 0}, {0, math.Inf(1)}, {-5, 0}, {0, 1e300}, {1e300, 0}}
	for i, v := range vs {
		for _, b := range bad {
			before := d.Clone()
			bt := b[0]
			if bt == 0 {
				bt = float64(i)
			}
			if _, err := d.Observe(bt, b[1]); err == nil {
				// Some huge values are in range early on; that is fine, then
				// this sample is simply accepted. Restore and go on.
				d = before
				continue
			}
			if !reflect.DeepEqual(d, before) {
				t.Fatalf("sample %d: rejected %v changed state", i, b)
			}
		}
		if _, err := d.Observe(float64(i), v); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFreshRunReportFinite(t *testing.T) {
	// A huge jump right after burn-in makes the newborn hypothesis the most
	// probable one: its report must still be finite.
	d, _ := New(WithBurnIn(3))
	for i, v := range []float64{0, 1, 0, 1e6, 1e6 + 1} {
		r, err := d.Observe(float64(i), v)
		if err != nil || !finiteReport(r) {
			t.Fatalf("sample %d: %+v %v", i, r, err)
		}
	}
}
