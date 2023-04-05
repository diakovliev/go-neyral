package network

import (
	"math"
	"math/rand"
	"testing"

	"github.com/diakovliev/go-neyral/common"
)

func shuffle[T any](data []T) (ret []T) {
	ret = data[:]
	rand.Shuffle(len(data), func(i, j int) {
		e := ret[i]
		ret[i] = ret[j]
		ret[j] = e
	})
	return
}

func TestNetwork(t *testing.T) {

	expectedError := 0.01
	speed := .9
	momentum := .2
	maxHoldCount := 10

	//net := NewNetwork(common.Sigma{}, 2).Init(true, 5, 3, 1)
	net := NewNetwork(common.Tanh{}, 2).Init(false, 10, 1)

	type teachData struct {
		In []float64
		Ex []float64
	}

	data := []teachData{
		// XOR
		{
			In: []float64{0., 1.},
			Ex: []float64{1.},
		},
		{
			In: []float64{0., 0.},
			Ex: []float64{0.},
		},
		{
			In: []float64{1., 0.},
			Ex: []float64{1.},
		},
		{
			In: []float64{1., 1.},
			Ex: []float64{0.},
		},
	}

	// Teach
	prevAvgErr := 0.
	haveToBreak := false
	holdCount := 0
	for i := 0; ; {
		avgErr := 0.
		for _, d := range shuffle(data) {
			out := net.Predict(d.In)
			err := net.Error(d.Ex)
			if math.IsNaN(err) {
				haveToBreak = true
				break
			}
			t.Logf("%d: in: %#v out: %#v ex: %#v", i, d.In, out, d.Ex)
			avgErr += err
			net = net.BackPropagate(speed, momentum, d.Ex)
		}
		if haveToBreak {
			break
		}
		avgErr /= float64(len(data))
		delta := avgErr - prevAvgErr
		if delta == 0. {
			holdCount++
		} else {
			holdCount = 0
		}
		if holdCount > maxHoldCount {
			t.Fatalf("network hold")
			break
		}
		t.Logf("%d: avg err: %g delta: %g", i, avgErr, delta)
		if avgErr <= expectedError {
			break
		}
		prevAvgErr = avgErr
		i++
	}

	// Results
	for _, d := range data {
		out := net.Predict(d.In)
		t.Logf("in: %#v out: %#v ex: %#v", d.In, out, d.Ex)
	}
}
