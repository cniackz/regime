package regime

import (
	"math/rand/v2"
	"strconv"
	"testing"
)

func BenchmarkObserve(b *testing.B) {
	for _, mr := range []int{128, 512, 2048} {
		b.Run("MaxRun="+strconv.Itoa(mr), func(b *testing.B) {
			d, _ := New(WithMaxRun(mr))
			rng := rand.New(rand.NewPCG(1, 1))
			vs := make([]float64, 4096)
			for i := range vs {
				vs[i] = rng.NormFloat64()
			}
			// Warm up so the hypothesis buffer is at its working size.
			for i := 0; i < 4*mr; i++ {
				d.Observe(float64(i), vs[i%len(vs)])
			}
			t0 := 4 * mr
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := d.Observe(float64(t0+i), vs[i%len(vs)]); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestObserveDoesNotAllocate(t *testing.T) {
	d, _ := New()
	for i := 0; i < 2000; i++ {
		d.Observe(float64(i), float64(i%7))
	}
	i := 2000
	if n := testing.AllocsPerRun(500, func() {
		d.Observe(float64(i), float64(i%7))
		i++
	}); n != 0 {
		t.Fatalf("Observe allocates %v per call", n)
	}
}
