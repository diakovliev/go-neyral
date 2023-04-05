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

func (n Network) Init(withActivationNeyrons bool, layersCapacities ...uint) *Network {
	n.L = make([]*Layer, 0)
	prevLayerCapacity := n.N
	for i := 0; i < len(layersCapacities); i++ {

		var l *Layer
		if i < len(layersCapacities)-1 && withActivationNeyrons {
			l = NewLayer(n.F).
				WithDimensions(uint(prevLayerCapacity), layersCapacities[i]).
				WithActivation().
				InitWeights()
		} else {
			l = NewLayer(n.F).
				WithDimensions(uint(prevLayerCapacity), layersCapacities[i]).
				InitWeights()
		}

		n.L = append(n.L, l)

		if i < len(layersCapacities)-1 && withActivationNeyrons {
			prevLayerCapacity = layersCapacities[i] + 1
		} else {
			prevLayerCapacity = layersCapacities[i]
		}
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

func (n Network) BackPropagate(speed float64, momentum float64, expectations []float64) *Network {
	last := len(n.L) - 1
	n.L[last] = n.L[last].SetExpectations(expectations)
	nextLayerDeltas := []float64{}
	for i := last; i >= 0; i-- {
		n.L[i] = n.L[i].UpdateWeights(speed, momentum, nextLayerDeltas)
		nextLayerDeltas = n.L[i].Deltas()
	}
	return &n
}

func (n Network) Error(expectations []float64) float64 {
	return n.L[len(n.L)-1].SetExpectations(expectations).Error()
}
