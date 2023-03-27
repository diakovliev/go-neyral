package common

import "math"

type Sigma struct{}

// Direct
func (Sigma) F(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Inversion
func (Sigma) I(x float64) float64 {
	return math.Log(-(x / (x - 1)))
}
