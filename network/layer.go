package network

import (
	"fmt"
	"math"

	"github.com/diakovliev/go-neyral/common"
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	F      common.Activation
	N      []*Neyron
	iCount uint
}

func NewLayer(F common.Activation) *Layer {
	return &Layer{
		F: F,
	}
}

func (l Layer) WithDimensions(icount, ncount uint) *Layer {
	l.iCount = icount
	l.N = make([]*Neyron, ncount)
	for i := 0; i < len(l.N); i++ {
		l.N[i] = NewNeyron(l.F).WithInputs(icount)
	}
	return &l
}

func (l Layer) InitWeights() *Layer {
	for i := 0; i < len(l.N); i++ {
		l.N[i] = l.N[i].InitWeights()
	}
	return &l
}

func (l Layer) W() *mat.Dense {
	w := mat.NewDense(len(l.N), int(l.iCount), nil)
	for i := 0; i < len(l.N); i++ {
		w.SetRow(i, l.N[i].W)
	}
	return w
}

func (l Layer) IW() *mat.Dense {
	return common.MatLeftInversion(l.W())
}

func (l Layer) Out() (ret []float64) {
	ret = make([]float64, len(l.N))
	for i := 0; i < len(l.N); i++ {
		ret[i] = l.N[i].Out
	}
	return
}

// Activate layer
func (l Layer) Activate(in []float64) *Layer {

	X := mat.NewDense(len(in), 1, in)

	S := &mat.Dense{}
	S.Mul(l.W(), X)

	Y := &mat.Dense{}
	Y.Apply(func(i int, _ int, v float64) float64 {
		l.N[i].In = in[:]
		l.N[i].Out = l.N[i].F.F(v)
		return l.N[i].Out
	}, S)

	return &l
}

// BackPropagate through layer
func (l Layer) BackPropagate(out []float64) (in []float64) {

	Y := mat.NewDense(len(out), 1, out)

	IY := &mat.Dense{}
	IY.Apply(func(i int, _ int, v float64) float64 {
		return l.N[i].F.I(v)
		//return v
	}, Y)

	OX := &mat.Dense{}
	OX.Mul(l.IW(), IY)

	CX := OX.ColView(0)
	in = make([]float64, CX.Len())

	for i := 0; i < CX.Len(); i++ {
		in[i] = CX.AtVec(i)
	}

	return
}

func (l Layer) SetExpectations(e []float64) *Layer {
	if len(e) != len(l.N) {
		panic(fmt.Errorf("expectations length: %d is not matches layer neyrons count: %d", len(e), len(l.N)))
	}
	for i := 0; i < len(l.N); i++ {
		l.N[i].E = e[i]
	}
	return &l
}

func (l Layer) UpdateWeights(speed float64) *Layer {
	for i := 0; i < len(l.N); i++ {
		l.N[i] = l.N[i].UpdateWeights(speed)
	}
	return &l
}

func (l Layer) Error() float64 {
	var sum float64
	for i := 0; i < len(l.N); i++ {
		sum += l.N[i].Error()
	}
	sum = math.Sqrt(sum)
	sum /= math.Sqrt(float64(len(l.N)))
	return sum
}
