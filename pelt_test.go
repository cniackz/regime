package regime

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// PELT (Killick, Fearnhead and Eckley 2012) is the offline upper bound in the
// bank: it sees the whole series and finds the segmentation that minimizes
// cost plus a penalty per changepoint,
//
//	F(t) = min over tau in R_t of F(tau) + C(y[tau:t]) + beta,
//
// pruning tau from R whenever F(tau) + C(y[tau:t]) > F(t). This measurement
// was added after the seals and was NOT part of the sealed prediction.

// peltSums holds prefix sums of a series for O(1) segment costs.
type peltSums struct{ n, t, tt, y, ty, yy []float64 }

func newPeltSums(vs []float64) peltSums {
	k := len(vs) + 1
	s := peltSums{make([]float64, k), make([]float64, k), make([]float64, k), make([]float64, k), make([]float64, k), make([]float64, k)}
	for i, v := range vs {
		x := float64(i)
		s.n[i+1] = s.n[i] + 1
		s.t[i+1] = s.t[i] + x
		s.tt[i+1] = s.tt[i] + x*x
		s.y[i+1] = s.y[i] + v
		s.ty[i+1] = s.ty[i] + x*v
		s.yy[i+1] = s.yy[i] + v*v
	}
	return s
}

// cost returns the cost of y[a:b]: the residual sum of squares of a mean
// (linear false) or of a least-squares line (linear true), which is -2
// log-likelihood up to a constant for Gaussian noise of sigma 1.
func (s peltSums) cost(a, b int, linear bool) float64 {
	n := s.n[b] - s.n[a]
	sy, syy := s.y[b]-s.y[a], s.yy[b]-s.yy[a]
	rss := syy - sy*sy/n
	if linear && n > 2 {
		st, stt, sty := s.t[b]-s.t[a], s.tt[b]-s.tt[a], s.ty[b]-s.ty[a]
		sxx := stt - st*st/n
		sxy := sty - st*sy/n
		if sxx > 0 {
			rss -= sxy * sxy / sxx
		}
	}
	return max(rss, 0)
}

// pelt returns the changepoints of vs: indices where a new segment starts.
func pelt(vs []float64, linear bool) []int {
	n := len(vs)
	params := 2.0 // a mean and a changepoint location
	if linear {
		params = 3 // a level, a slope and a location
	}
	beta := params * math.Log(float64(n))
	minSeg := 2
	if linear {
		minSeg = 3
	}
	s := newPeltSums(vs)
	f := make([]float64, n+1)
	last := make([]int, n+1)
	f[0] = -beta
	cands := []int{0}
	for t := minSeg; t <= n; t++ {
		best, arg := math.Inf(1), 0
		for _, tau := range cands {
			if t-tau < minSeg {
				continue
			}
			if v := f[tau] + s.cost(tau, t, linear) + beta; v < best {
				best, arg = v, tau
			}
		}
		f[t], last[t] = best, arg
		keep := cands[:0]
		for _, tau := range cands {
			if t-tau < minSeg || f[tau]+s.cost(tau, t, linear) <= f[t] {
				keep = append(keep, tau)
			}
		}
		cands = append(keep, t-minSeg+1)
	}
	var cps []int
	for t := n; t > 0; t = last[t] {
		if last[t] > 0 {
			cps = append(cps, last[t])
		}
	}
	for i, j := 0, len(cps)-1; i < j; i, j = i+1, j-1 {
		cps[i], cps[j] = cps[j], cps[i]
	}
	return cps
}

func TestPeltFindsPlantedChanges(t *testing.T) {
	vs := noisy(21, 900, func(i int) float64 { return float64(i/300) * 3 })
	if cps := pelt(vs, false); len(cps) != 2 || absInt(cps[0]-300) > 3 || absInt(cps[1]-600) > 3 {
		t.Fatalf("pelt = %v, want about [300 600]", cps)
	}
	bendy := noisy(22, 600, func(i int) float64 { return bend(i, 0) })
	if cps := pelt(bendy, true); len(cps) != 1 || absInt(cps[0]-300) > 15 {
		t.Fatalf("pelt linear = %v, want about [300]", cps)
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// TestBankPELT compares the offline PELT segmentation with regime's online
// estimate of the start, Since, on the alarm sample. Same series and seed as
// TestBank. Not sealed.
func TestBankPELT(t *testing.T) {
	if os.Getenv("REGIME_BANK") != "1" {
		t.Skip("set REGIME_BANK=1 to run the synthetic bank")
	}
	var out strings.Builder
	pf := func(format string, a ...any) { fmt.Fprintf(&out, format, a...) }
	pf("Offline upper bound, PELT (Killick, Fearnhead and Eckley 2012), penalty p*log(n) with sigma 1 known. ")
	pf("Added after the seals: NOT part of the sealed prediction.\n\n")
	pf("| Case | Method | Change found within 5 samples of tau | Median abs error of the start | Spurious changes / 1000 samples |\n")
	pf("|---|---|---|---|---|\n")
	for _, c := range []bankCase{caseC1, caseC2, caseC3, caseC4, caseHalf} {
		linear := c.id == caseC4.id
		type res struct {
			found     bool
			err       float64
			spurious  int
			rFound    bool
			rErr      float64
			rSpurious int
		}
		rs := parallelMap(bankCount, func(s int) res {
			vs := bankSeries(c, s)
			var r res
			best := math.Inf(1)
			for _, cp := range pelt(vs, linear) {
				e := math.Abs(float64(cp - bankTau))
				if c.id == caseC1.id || e > bankHorizn {
					r.spurious++
					continue
				}
				best = math.Min(best, e)
			}
			if c.id != caseC1.id && !math.IsInf(best, 1) {
				r.err = best
				r.found = best <= 5
			} else {
				r.err = math.NaN()
			}
			// regime: Since on the first alarm in [tau, tau + horizon].
			d, _ := New()
			r.rErr = math.NaN()
			first := true
			for i, v := range vs {
				rep, _ := d.Observe(float64(i), v)
				if !rep.Alarm {
					continue
				}
				if c.id == caseC1.id || i < bankTau || !first || i-bankTau > bankHorizn {
					r.rSpurious++
					continue
				}
				first = false
				r.rErr = math.Abs(rep.Since - bankTau)
				r.rFound = r.rErr <= 5
			}
			return r
		})
		var pf1, rf1, ps, rsp int
		var pe, re []float64
		for _, r := range rs {
			if r.found {
				pf1++
			}
			if r.rFound {
				rf1++
			}
			ps += r.spurious
			rsp += r.rSpurious
			if !math.IsNaN(r.err) {
				pe = append(pe, r.err)
			}
			if !math.IsNaN(r.rErr) {
				re = append(re, r.rErr)
			}
		}
		samples := float64(bankCount * c.n)
		row := func(m string, found int, errs []float64, sp int) {
			fs, es := "n/a", "n/a"
			if c.id != caseC1.id {
				fs = fmt.Sprintf("%.1f%%", 100*float64(found)/bankCount)
				es = fmt.Sprintf("%.1f", medianOf(errs))
			}
			pf("| %s | %s | %s | %s | %.3f |\n", c.name, m, fs, es, 1000*float64(sp)/samples)
		}
		row("PELT, offline", pf1, pe, ps)
		row("regime, online", rf1, re, rsp)
	}
	pf("\nregime counts as spurious any alarm other than the first one within the horizon after tau.\n")
	t.Log("\n" + out.String())
	if path := os.Getenv("REGIME_BANK_OUT"); path != "" {
		if err := os.WriteFile(path, []byte(out.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
