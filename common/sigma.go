package common

import (
	"fmt"
	"math"
)

type Sigma struct{}

// Direct
func (Sigma) F(x float64) (ret float64) {
	if x <= -9.999999 {
		return 0.
	}
	if x >= 9.999999 {
		return 1.
	}
	ret = 1. / (1. + math.Exp(-x))
	if math.IsNaN(ret) {
		panic(fmt.Errorf("F nan x: %g", x))
	}
	return
}

// Inversion
func (Sigma) I(x float64) (ret float64) {
	if x <= 0. {
		return -9.999999
	}
	if x >= 1.0 {
		return 9.999999
	}
	ret = math.Log(-x / (x - 1))
	if math.IsNaN(ret) {
		panic(fmt.Errorf("I nan x: %g", x))
	}
	return
}

// Prime
func (Sigma) P(x float64) float64 {
	return x * (1 - x)
}
