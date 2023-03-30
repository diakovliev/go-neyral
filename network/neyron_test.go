package network

import (
	"testing"

	"github.com/diakovliev/go-neyral/common"
)

func TestNeyron(t *testing.T) {
	n := NewNeyron(&common.Sigma{}).WithInputs(2).InitWeights()

	_ = n.Activate([]float64{0.5, 0.5}).Out
}
