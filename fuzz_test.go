package regime

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

// FuzzObserve feeds arbitrary float64 pairs, the full class (NaN, infinities,
// subnormals, huge values, equal and decreasing times): no panic, every
// accepted report finite, and a rejected sample leaves the state unchanged.
func FuzzObserve(f *testing.F) {
	seed := func(xs ...float64) []byte {
		b := make([]byte, 0, 8*len(xs))
		for _, x := range xs {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(x))
		}
		return b
	}
	f.Add(uint8(0), seed(0, 1, 1, 2, 2, 3, 3, 1e300, 4, math.NaN(), 5, 5e-324))
	f.Add(uint8(1), seed(1, 1, 1, 1, 0, 0, math.Inf(1), 0, 2, -1e308, 3, 1e308))
	f.Fuzz(func(t *testing.T, opt uint8, data []byte) {
		opts := []Option{WithBurnIn(2 + int(opt%5)), WithMaxRun(2 + int(opt%7)), WithMinRun(1 + int(opt%3))}
		if opt&0x80 != 0 {
			opts = append(opts, WithModel(Level))
		}
		d, err := New(opts...)
		if err != nil {
			t.Fatal(err)
		}
		for len(data) >= 16 {
			tt := math.Float64frombits(binary.LittleEndian.Uint64(data))
			v := math.Float64frombits(binary.LittleEndian.Uint64(data[8:]))
			data = data[16:]
			before := d.Clone()
			r, err := d.Observe(tt, v)
			if err != nil {
				if !reflect.DeepEqual(d, before) {
					t.Fatalf("Observe(%v, %v) = %v changed state", tt, v, err)
				}
				continue
			}
			if !finiteReport(r) {
				t.Fatalf("Observe(%v, %v): non-finite %+v", tt, v, r)
			}
		}
	})
}

// FuzzNew tries arbitrary option values: New either fails with ErrOption or
// returns a detector that accepts samples.
func FuzzNew(f *testing.F) {
	f.Add(int8(0), 250.0, 512, 0.5, 10, 30, 10)
	f.Fuzz(func(t *testing.T, m int8, er float64, mr int, th float64, w, b, mn int) {
		if mr > 1<<12 || b > 1<<12 {
			t.Skip("allocation too large for a fuzz iteration")
		}
		d, err := New(WithModel(Model(m)), WithExpectedRun(er), WithMaxRun(mr),
			WithThreshold(th), WithWindow(w), WithBurnIn(b), WithMinRun(mn))
		if err != nil {
			if err != ErrOption {
				t.Fatal(err)
			}
			return
		}
		for i := 0; i < 2*b+5; i++ {
			r, err := d.Observe(float64(i), float64(i%3))
			if err != nil || !finiteReport(r) {
				t.Fatalf("sample %d: %+v %v", i, r, err)
			}
		}
	})
}
