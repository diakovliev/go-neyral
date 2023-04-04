package network

import (
	"math"
	"testing"

	"github.com/diakovliev/go-neyral/common"
)

func TestNetwork(t *testing.T) {

	expectedError := 0.001
	speed := .9

	net := NewNetwork(common.Sigma{}, 2).Init(2, 1)

	type teachData struct {
		In []float64
		Ex []float64
	}

	data := []teachData{
		// OR
		{
			In: []float64{0., 1.},
			Ex: []float64{1.},
		},
		{
			In: []float64{1., 0.},
			Ex: []float64{1.},
		},
		{
			In: []float64{0., 0.},
			Ex: []float64{0.},
		},
		{
			In: []float64{1., 1.},
			Ex: []float64{1.},
		},
	}

	// Teach
	prevAvgErr := 0.
	haveToBreak := false
	for i := 0; ; {
		avgErr := 0.
		for _, d := range data {
			out := net.Predict(d.In)
			err := net.Error(d.Ex)
			if math.IsNaN(err) {
				haveToBreak = true
				break
			}
			t.Logf("%d: in: %#v out: %#v ex: %#v", i, d.In, out, d.Ex)
			avgErr += err
			net = net.BackPropagate(speed, d.Ex)
		}
		if haveToBreak {
			break
		}
		avgErr /= float64(len(data))
		t.Logf("%d: avg err: %g delta: %g", i, avgErr, prevAvgErr-avgErr)
		if avgErr <= expectedError {
			break
		}
		prevAvgErr = avgErr
		i++
	}

	// Results
	for _, d := range data {
		out := net.Predict(d.In)
		err := net.Error(d.Ex)
		t.Logf("out: %#v err: %g", out, err)
	}
}
