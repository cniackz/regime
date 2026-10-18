package regime

import (
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestNewOptions(t *testing.T) {
	cases := []struct {
		name string
		opt  Option
		ok   bool
	}{
		{"default", nil, true},
		{"level", WithModel(Level), true},
		{"bad model", WithModel(Model(7)), false},
		{"expected run 1", WithExpectedRun(1), false},
		{"expected run NaN", WithExpectedRun(math.NaN()), false},
		{"expected run Inf", WithExpectedRun(math.Inf(1)), false},
		{"expected run 2", WithExpectedRun(2), true},
		{"max run 1", WithMaxRun(1), false},
		{"max run 2", WithMaxRun(2), true},
		{"max run huge", WithMaxRun(1 << 17), false},
		{"threshold 0", WithThreshold(0), false},
		{"threshold 1", WithThreshold(1), true},
		{"threshold NaN", WithThreshold(math.NaN()), false},
		{"window 0", WithWindow(0), false},
		{"window 1", WithWindow(1), true},
		{"burn-in 1", WithBurnIn(1), false},
		{"burn-in 2", WithBurnIn(2), true},
		{"min run 0", WithMinRun(0), false},
		{"min run 1", WithMinRun(1), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := New(c.opt)
			if c.ok != (err == nil) {
				t.Fatalf("err = %v, want ok=%v", err, c.ok)
			}
			if err != nil && !errors.Is(err, ErrOption) {
				t.Fatalf("err = %v, want ErrOption", err)
			}
			if err == nil && d == nil {
				t.Fatal("nil detector")
			}
		})
	}
}

func TestModelString(t *testing.T) {
	if Trend.String() != "Trend" || Level.String() != "Level" || Model(9).String() != "Model(invalid)" {
		t.Fatal("bad String")
	}
	var m Model
	if m != Trend {
		t.Fatal("zero Model is not Trend")
	}
}

func noisy(seed uint64, n int, f func(i int) float64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 1))
	vs := make([]float64, n)
	for i := range vs {
		vs[i] = f(i) + rng.NormFloat64()
	}
	return vs
}

func feed(t *testing.T, d *Detector, vs []float64) []Report {
	t.Helper()
	out := make([]Report, len(vs))
	for i, v := range vs {
		r, err := d.Observe(float64(i), v)
		if err != nil {
			t.Fatalf("sample %d: %v", i, err)
		}
		out[i] = r
	}
	return out
}

func TestZeroValueMatchesNew(t *testing.T) {
	vs := noisy(3, 400, func(i int) float64 {
		if i >= 200 {
			return 3
		}
		return 0
	})
	var z Detector
	d, _ := New()
	a, b := feed(t, &z, vs), feed(t, d, vs)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("zero value and New() differ")
	}
}

func TestErrors(t *testing.T) {
	d, _ := New()
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := d.Observe(0, v); !errors.Is(err, ErrNonFinite) {
			t.Fatalf("v=%v: err = %v", v, err)
		}
		if _, err := d.Observe(v, 0); !errors.Is(err, ErrNonFinite) {
			t.Fatalf("t=%v: err = %v", v, err)
		}
	}
	if _, err := d.Observe(5, 1); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []float64{5, 4, -1} {
		if _, err := d.Observe(tt, 1); !errors.Is(err, ErrTimeOrder) {
			t.Fatalf("t=%v: err = %v", tt, err)
		}
	}
	if r, _ := d.Observe(6, 1); r.N != 2 {
		t.Fatalf("N = %d, want 2", r.N)
	}
}

func TestBurnIn(t *testing.T) {
	d, _ := New(WithBurnIn(5))
	vs := []float64{1, 2, 3, 4, 5, 100, 100}
	for i, v := range vs {
		r, err := d.Observe(float64(i), v)
		if err != nil {
			t.Fatal(err)
		}
		if i < 5 && (r.Alarm || r.ChangeProb != 0) {
			t.Fatalf("sample %d in burn-in: %+v", i, r)
		}
		if r.N != i+1 {
			t.Fatalf("N = %d", r.N)
		}
	}
}

func TestResetAndClone(t *testing.T) {
	vs := noisy(4, 300, func(i int) float64 { return 0.01 * float64(i) })
	d, _ := New()
	first := feed(t, d, vs)
	d.Reset()
	if again := feed(t, d, vs); !reflect.DeepEqual(first, again) {
		t.Fatal("Reset does not reproduce")
	}
	d.Reset()
	feed(t, d, vs[:150])
	c := d.Clone()
	for i := 150; i < 300; i++ {
		a, _ := d.Observe(float64(i), vs[i])
		b, _ := c.Observe(float64(i), vs[i])
		if a != b {
			t.Fatalf("clone diverged at %d", i)
		}
	}
	// Mutating the clone must not touch the original.
	c.Observe(1000, 1e6)
	if d.n != 300 || c.n != 301 {
		t.Fatal("clone shares state")
	}
	var z Detector
	if z.Clone().ready {
		t.Fatal("clone of zero value built buffers")
	}
	z.Reset()
}

func TestLgammaTable(t *testing.T) {
	d, _ := New()
	for n := range d.lgc {
		nu := 2*priorAlpha + float64(n)
		if got, want := d.lgc[n], lgammaRatio(nu); got != want {
			t.Fatalf("n=%d: %v != %v", n, got, want)
		}
	}
	if got, want := d.lgammaRatio(len(d.lgc)+3, 2*priorAlpha+float64(len(d.lgc)+3)), lgammaRatio(2*priorAlpha+float64(len(d.lgc)+3)); got != want {
		t.Fatal("fallback differs")
	}
	// logStudentT integrates to about 1.
	var sum float64
	for y := -60.0; y <= 60; y += 0.001 {
		sum += math.Exp(logStudentT(y, 0.5, 2, 5)) * 0.001
	}
	if math.Abs(sum-1) > 1e-3 {
		t.Fatalf("density integrates to %v", sum)
	}
}
