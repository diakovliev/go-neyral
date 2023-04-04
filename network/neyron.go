package network

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/diakovliev/go-neyral/common"
	"gonum.org/v1/gonum/mat"
)

type Neyron struct {
	F common.Activation
	// Weights
	W []float64
	D []float64
	// Inputs
	In []float64
	// Output
	Out float64
	// Expectation
	E float64
}

func NewNeyron(F common.Activation) *Neyron {
	return &Neyron{F: F}
}

func (n Neyron) WithInputs(count uint) *Neyron {
	n.In = make([]float64, count)
	return &n
}

func (n Neyron) InitWeights() *Neyron {
	n.W = make([]float64, len(n.In))
	for i := 0; i < len(n.In); i++ {
		n.W[i] = rand.NormFloat64()
	}
	return &n
}

func (n Neyron) Activate(in []float64) *Neyron {

	if len(in) != len(n.In) {
		panic(fmt.Errorf("input size mismatch: got: %d expected: %d", len(in), len(n.In)))
	}

	n.In = in[:]

	vin := mat.NewVecDense(len(n.In), n.In)
	vw := mat.NewVecDense(len(n.W), n.W)

	n.Out = n.F.F(mat.Dot(vin, vw))

	return &n
}

func (n Neyron) UpdateWeights(speed float64) *Neyron {
	momentum := 0.5
	n.D = make([]float64, len(n.W))
	for i := 0; i < len(n.W); i++ {
		delta := speed*n.E*n.F.P(n.Out)*n.In[i] + momentum*n.D[i]
		n.W[i] += delta
		n.D[i] = delta
	}
	return &n
}

func (n Neyron) Error() float64 {
	return math.Pow(n.E, 2)
}
