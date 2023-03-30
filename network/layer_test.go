package network

import (
	"testing"

	"github.com/diakovliev/go-neyral/common"
)

func TestLayerLearn(t *testing.T) {

	l := NewLayer(&common.Sigma{}).WithDimensions( /*inputs count*/ 2 /*outputs (neurons) count*/, 3).InitWeights()

	speed := 1.
	in := []float64{1.0, 0.0}
	ex := []float64{0.2, 0.3, 0.4}
	goal := 0.001

	out := l.Activate(in).Out()

	t.Logf("in: %#v", in)
	t.Logf("ex: %#v", ex)

	t.Logf("out: %#v", out)

	tl := l
	var i uint
	for {
		tl = tl.SetExpectations(ex)

		if tl.Error() <= goal {
			break
		}

		t.Logf("%d: err: %#v", i, tl.Error())

		tl = tl.UpdateWeights(speed)
		out = tl.Activate(in).Out()

		t.Logf("%d: out: %#v", i, out)
		i++
	}

	t.Logf("------------------ \nerr: %#v", tl.Error())
	out = tl.Activate(in).Out()

	t.Logf("out: %#v", out)
}
