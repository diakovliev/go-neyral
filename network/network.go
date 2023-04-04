package network

import "github.com/diakovliev/go-neyral/common"

type Network struct {
	F common.Activation
	N uint
	L []*Layer
}

func NewNetwork(f common.Activation, ninputs uint) *Network {
	return &Network{
		F: f,
		N: ninputs,
	}
}

func (n Network) Init(layersCapacities ...uint) *Network {
	n.L = make([]*Layer, 0)
	prevLayerCapacity := n.N
	for i := 0; i < len(layersCapacities); i++ {
		n.L = append(n.L, NewLayer(n.F).
			WithDimensions(uint(prevLayerCapacity), layersCapacities[i]).
			InitWeights())
	}
	return &n
}

func (n Network) Predict(in []float64) (out []float64) {
	out = in
	for i := 0; i < len(n.L); i++ {
		n.L[i] = n.L[i].Activate(out)
		out = n.L[i].Out()
	}
	return
}

func (n Network) BackPropagate(speed float64, expectations []float64) *Network {
	errs := expectations
	last := len(n.L) - 1
	for i := last; i >= 0; i-- {
		if last == i {
			n.L[i] = n.L[i].SetExpectations(errs)
		} else {
			n.L[i] = n.L[i].SetErrors(errs)
		}
		n.L[i] = n.L[i].UpdateWeights(speed)
		errs = n.L[i].BackPropagate(errs)
	}
	for i := len(n.L) - 1; i >= 0; i-- {
		//n.L[i] = n.L[i].UpdateWeights(speed)
	}
	return &n
}

func (n Network) Error(expectations []float64) float64 {
	return n.L[len(n.L)-1].SetExpectations(expectations).Error()
}
