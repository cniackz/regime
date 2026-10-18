package regime

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
)

// The synthetic bank measures the sealed claims (README, "Measured results").
// It is slow, so it runs only with REGIME_BANK=1:
//
//	REGIME_BANK=1 go test -run TestBank -timeout 1h -v .
//
// Every series comes from a fixed seed, so the numbers are reproducible.

const (
	bankSeed   = 20261018
	bankTau    = 300
	bankLen    = 600
	bankFlat   = 1000 // length of the no-change series
	bankCount  = 1000 // series per case
	bankHorizn = 50   // a detection must come within tau + 50
)

type bankCase struct {
	id    int
	name  string
	n     int
	shape func(i, s int) float64 // noise-free value of series s at sample i
}

func levelShift(delta float64) func(i, s int) float64 {
	return func(i, s int) float64 {
		if i < bankTau {
			return 0
		}
		if s%2 == 1 {
			return -delta
		}
		return delta
	}
}

func bend(i, _ int) float64 {
	if i < bankTau {
		return 0
	}
	return 0.1 * float64(i-bankTau)
}

var (
	caseC1   = bankCase{1, "C1 no change", bankFlat, func(int, int) float64 { return 0 }}
	caseC2   = bankCase{2, "C2 shift 2 sigma", bankLen, levelShift(2)}
	caseC3   = bankCase{3, "C3 shift 4 sigma", bankLen, levelShift(4)}
	caseC4   = bankCase{4, "C4 bend 0.1 sigma/sample", bankLen, bend}
	caseHalf = bankCase{6, "C6 shift 0.5 sigma", bankLen, levelShift(0.5)}
)

func bankSeries(c bankCase, s int) []float64 {
	rng := rand.New(rand.NewPCG(bankSeed, uint64(c.id)<<32|uint64(s)))
	vs := make([]float64, c.n)
	for i := range vs {
		vs[i] = c.shape(i, s) + rng.NormFloat64()
	}
	return vs
}

// seriesResult is what one detector run on one series produced.
type seriesResult struct {
	falseAlarms int     // alarms before tau, or any alarm with no change
	detected    bool    // first alarm at or after tau lies within the horizon
	delay       int     // that alarm's index minus tau
	covered     bool    // SinceLow <= tau <= SinceHigh on the alarm sample
	sinceBias   float64 // Since - tau on the alarm sample
}

func runRegime(c bankCase, vs []float64, plain bool) seriesResult {
	cfg := defaultConfig()
	cfg.plainPrior = plain
	d := &Detector{cfg: cfg}
	var res seriesResult
	first := true
	for i, v := range vs {
		r, err := d.Observe(float64(i), v)
		if err != nil {
			panic(err)
		}
		if !r.Alarm {
			continue
		}
		if c.id == 1 || i < bankTau {
			res.falseAlarms++
			continue
		}
		if first {
			first = false
			if i-bankTau <= bankHorizn {
				res.detected = true
				res.delay = i - bankTau
				res.covered = r.SinceLow <= bankTau && bankTau <= r.SinceHigh
				res.sinceBias = r.Since - bankTau
			}
		}
	}
	return res
}

// cusum is a two-sided CUSUM (Page 1954) given the true pre-change mean 0 and
// sigma 1, tuned for a shift of 2 sigma (reference value k = 1), restarted
// after each alarm. It returns the alarm indices.
func cusum(vs []float64, h float64) []int {
	const k = 1.0
	var hi, lo float64
	var out []int
	for i, v := range vs {
		hi = max(0, hi+v-k)
		lo = max(0, lo-v-k)
		if hi > h || lo > h {
			out = append(out, i)
			hi, lo = 0, 0
		}
	}
	return out
}

func parallelMap[T any](n int, f func(s int) T) []T {
	out := make([]T, n)
	var wg sync.WaitGroup
	next := make(chan int)
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := range next {
				out[s] = f(s)
			}
		}()
	}
	for s := range n {
		next <- s
	}
	close(next)
	wg.Wait()
	return out
}

type caseSummary struct {
	faPer1000   float64
	detected    float64 // fraction
	within      float64 // fraction detected with delay <= the case's bound
	median      float64 // median delay among detected
	coverage    float64 // fraction of detected cases covered
	sinceBias   float64 // median Since - tau among detected
	censoredMed float64 // median delay, undetected counted as horizon + 1
}

func summarize(c bankCase, rs []seriesResult, bound int) caseSummary {
	var s caseSummary
	var fa, det, within, cov int
	var delays, biases, cens []float64
	for _, r := range rs {
		fa += r.falseAlarms
		if r.detected {
			det++
			delays = append(delays, float64(r.delay))
			biases = append(biases, r.sinceBias)
			cens = append(cens, float64(r.delay))
			if r.delay <= bound {
				within++
			}
			if r.covered {
				cov++
			}
		} else {
			cens = append(cens, bankHorizn+1)
		}
	}
	samples := len(rs) * c.n
	s.faPer1000 = 1000 * float64(fa) / float64(samples)
	s.detected = float64(det) / float64(len(rs))
	s.within = float64(within) / float64(len(rs))
	s.median = medianOf(delays)
	s.sinceBias = medianOf(biases)
	s.censoredMed = medianOf(cens)
	if det > 0 {
		s.coverage = float64(cov) / float64(det)
	}
	return s
}

func medianOf(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := slices.Clone(xs)
	slices.Sort(s)
	return median(s)
}

// cusumDelays returns the censored median delay of CUSUM on a case.
func cusumDelays(c bankCase, series [][]float64, h float64) float64 {
	var cens []float64
	for _, vs := range series {
		d := float64(bankHorizn + 1)
		for _, a := range cusum(vs, h) {
			if a >= bankTau {
				if a-bankTau <= bankHorizn {
					d = float64(a - bankTau)
				}
				break
			}
		}
		cens = append(cens, d)
	}
	return medianOf(cens)
}

func cusumRate(series [][]float64, h float64) float64 {
	var n, fa int
	for _, vs := range series {
		n += len(vs)
		fa += len(cusum(vs, h))
	}
	return 1000 * float64(fa) / float64(n)
}

func TestBank(t *testing.T) {
	if os.Getenv("REGIME_BANK") != "1" {
		t.Skip("set REGIME_BANK=1 to run the synthetic bank")
	}
	var out strings.Builder
	pf := func(format string, a ...any) { fmt.Fprintf(&out, format, a...) }

	cases := []struct {
		c     bankCase
		bound int
	}{{caseC1, 0}, {caseC2, 1000}, {caseC3, 5}, {caseC4, 30}}
	series := map[int][][]float64{}
	for _, cc := range append(cases, struct {
		c     bankCase
		bound int
	}{caseHalf, 1000}) {
		series[cc.c.id] = parallelMap(bankCount, func(s int) []float64 { return bankSeries(cc.c, s) })
	}

	sum := map[bool]map[int]caseSummary{true: {}, false: {}}
	for _, plain := range []bool{false, true} {
		for _, cc := range cases {
			ss := series[cc.c.id]
			rs := parallelMap(bankCount, func(s int) seriesResult { return runRegime(cc.c, ss[s], plain) })
			sum[plain][cc.c.id] = summarize(cc.c, rs, cc.bound)
		}
	}
	half := parallelMap(bankCount, func(s int) seriesResult { return runRegime(caseHalf, series[caseHalf.id][s], false) })
	halfSum := summarize(caseHalf, half, 1000)

	e := sum[false]
	p := sum[true]
	pf("| Case | Prior | False alarms / 1000 samples | Detected | Within bound | Median delay | Coverage | Median Since - tau |\n")
	pf("|---|---|---|---|---|---|---|---|\n")
	for _, cc := range cases {
		for _, plain := range []bool{false, true} {
			s := sum[plain][cc.c.id]
			name := "empirical"
			if plain {
				name = "plain"
			}
			pf("| %s | %s | %.3f | %.1f%% | %.1f%% | %.1f | %.1f%% | %.1f |\n", cc.c.name, name,
				s.faPer1000, 100*s.detected, 100*s.within, s.median, 100*s.coverage, s.sinceBias)
		}
	}
	pf("| %s | empirical | %.3f | %.1f%% | | %.1f | %.1f%% | %.1f |\n\n", caseHalf.name,
		halfSum.faPer1000, 100*halfSum.detected, halfSum.median, 100*halfSum.coverage, halfSum.sinceBias)

	// C6: CUSUM threshold matched to the measured C1 rate of regime. With a
	// rate of 0, take the smallest threshold with no false alarm on C1.
	target := e[1].faPer1000
	lo, hi := 0.0, 50.0
	for range 60 {
		mid := (lo + hi) / 2
		if cusumRate(series[1], mid) > target {
			lo = mid
		} else {
			hi = mid
		}
	}
	h := hi
	pf("CUSUM (k = 1, true pre-change mean and sigma given), h = %.3f, C1 rate %.3f / 1000 (regime %.3f)\n\n",
		h, cusumRate(series[1], h), target)
	pf("| Shift | regime censored median delay | CUSUM censored median delay |\n|---|---|---|\n")
	for _, x := range []struct {
		name string
		id   int
		s    caseSummary
	}{{"0.5 sigma", caseHalf.id, halfSum}, {"2 sigma", 2, e[2]}, {"4 sigma", 3, e[3]}} {
		pf("| %s | %.1f | %.1f |\n", x.name, x.s.censoredMed, cusumDelays(bankCase{}, series[x.id], h))
	}
	pf("\n")

	lorden := math.Log(1000) / 2
	verdict := func(ok bool) string {
		if ok {
			return "PASS"
		}
		return "FAIL"
	}
	cusHalf := cusumDelays(caseHalf, series[caseHalf.id], h)
	cus2 := cusumDelays(caseC2, series[2], h)
	cus4 := cusumDelays(caseC3, series[3], h)
	ratio := e[1].faPer1000 / math.Max(p[1].faPer1000, 1e-12)
	inBand := func(x float64) bool { return x >= 0.80 && x <= 0.98 }
	pf("Claims (sealed):\n\n")
	pf("1. C1 <= 1 false alarm / 1000: %.3f, %s\n", e[1].faPer1000, verdict(e[1].faPer1000 <= 1))
	pf("2. C2 >= 95%% detected and median delay <= %.1f: %.1f%%, %.1f, %s\n", 2*lorden, 100*e[2].detected, e[2].median,
		verdict(e[2].detected >= 0.95 && e[2].median <= 2*lorden))
	pf("3. C3 >= 99%% within 5: %.1f%%, %s\n", 100*e[3].within, verdict(e[3].within >= 0.99))
	pf("4. C4 >= 90%% within 30: %.1f%%, %s\n", 100*e[4].within, verdict(e[4].within >= 0.90))
	pf("5. coverage in [80%%, 98%%]: C2 %.1f%%, C3 %.1f%%, C4 %.1f%%, %s\n", 100*e[2].coverage, 100*e[3].coverage,
		100*e[4].coverage, verdict(inBand(e[2].coverage) && inBand(e[3].coverage) && inBand(e[4].coverage)))
	pf("6. CUSUM faster at 2 sigma (%.1f vs %.1f), regime faster at 0.5 (%.1f vs %.1f) and 4 sigma (%.1f vs %.1f): %s\n",
		cus2, e[2].censoredMed, halfSum.censoredMed, cusHalf, e[3].censoredMed, cus4,
		verdict(cus2 < e[2].censoredMed && halfSum.censoredMed < cusHalf && e[3].censoredMed < cus4))
	pf("7. empirical C1 rate >= plain: %.3f vs %.3f, %s\n", e[1].faPer1000, p[1].faPer1000, verdict(e[1].faPer1000 >= p[1].faPer1000))
	pf("8. C1 rates within a factor of 2: ratio %.2f, %s\n", ratio, verdict(e[1].faPer1000 == p[1].faPer1000 || (ratio < 2 && ratio > 0.5)))
	ok9 := true
	for _, id := range []int{2, 3, 4} {
		ok9 = ok9 && math.Abs(e[id].sinceBias) <= 3 && math.Abs(p[id].sinceBias) <= 3
	}
	pf("9. |median Since bias| <= 3, both priors: empirical %.1f/%.1f/%.1f, plain %.1f/%.1f/%.1f, %s\n",
		e[2].sinceBias, e[3].sinceBias, e[4].sinceBias, p[2].sinceBias, p[3].sinceBias, p[4].sinceBias, verdict(ok9))
	maxPP := 0.0
	for _, id := range []int{2, 3, 4} {
		maxPP = math.Max(maxPP, math.Abs(e[id].coverage-p[id].coverage))
	}
	pf("Original check: empirical prior moves coverage by < 10 points: max %.1f points, %s\n", 100*maxPP, verdict(maxPP < 0.10))

	t.Log("\n" + out.String())
	if path := os.Getenv("REGIME_BANK_OUT"); path != "" {
		if err := os.WriteFile(path, []byte(out.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
