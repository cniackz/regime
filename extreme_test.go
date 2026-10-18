package regime

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExtremeInputsNeverPanic mixes raw float64 bit patterns, magnitudes from
// 1e-100 to 1e100 and jumping times. It caught a run where rounding made every
// hypothesis mass NaN, so prune dropped them all and the report indexed an
// empty slice.
func TestExtremeInputsNeverPanic(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	pick := func() float64 {
		switch rng.IntN(6) {
		case 0:
			return math.Float64frombits(rng.Uint64())
		case 1:
			return math.Pow(10, rng.Float64()*200-100) * float64(1-2*rng.IntN(2))
		case 2:
			return float64(rng.IntN(5))
		}
		return rng.NormFloat64()
	}
	for it := 0; it < 100000; it++ {
		d, _ := New(WithBurnIn(2+rng.IntN(4)), WithMaxRun(2+rng.IntN(6)), WithModel(Model(rng.IntN(2))))
		tt := 0.0
		for k := 0; k < 12; k++ {
			if rng.IntN(3) == 0 {
				tt = pick()
			} else {
				tt += math.Abs(pick())
			}
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("panic it=%d: %v", it, r)
					}
				}()
				r, err := d.Observe(tt, pick())
				if err == nil && !finiteReport(r) {
					t.Fatalf("nonfinite %+v", r)
				}
			}()
		}
	}
}
