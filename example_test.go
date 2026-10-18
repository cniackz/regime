package regime_test

import (
	"fmt"

	"github.com/cniackz/regime"
)

// A level jumps from 10 to 14 at t = 60 in a series with a small ripple.
func Example() {
	d, err := regime.New(regime.WithModel(regime.Level))
	if err != nil {
		panic(err)
	}
	ripple := []float64{0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.1, 0.4, -0.3, 0.2}
	var last regime.Report
	for i := 0; i < 120; i++ {
		v := 10 + ripple[i%len(ripple)]
		if i >= 60 {
			v += 4
		}
		r, err := d.Observe(float64(i), v)
		if err != nil {
			panic(err)
		}
		if r.Alarm {
			fmt.Printf("t=%d: new regime since %v\n", i, r.Since)
		}
		last = r
	}
	fmt.Printf("level %.1f over the last %d samples\n", last.Level, last.RunLength)
	// Output:
	// t=60: new regime since 60
	// level 14.0 over the last 60 samples
}

// The zero value works with the defaults (the Trend model).
func ExampleDetector_Observe() {
	var d regime.Detector
	for i := 0; i < 100; i++ {
		v := 0.5 * float64(i) // a clean ramp, slope 0.5 per unit of t
		if i >= 50 {
			v = 25 + 2*float64(i-50) // the slope bends to 2
		}
		r, _ := d.Observe(float64(i), v+0.01*float64(i%3))
		if r.Alarm {
			fmt.Printf("t=%d: alarm\n", i)
		}
		if i == 99 {
			fmt.Printf("since %v, slope %.2f\n", r.Since, r.Slope)
		}
	}
	// Output:
	// t=54: alarm
	// since 51, slope 2.00
}
