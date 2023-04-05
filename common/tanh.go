package common

import "math"

type Tanh struct{}

const tanhLimit = 9.9999999999999999999999999999999

func (Tanh) F(x float64) float64 {
	if x <= -tanhLimit {
		return -1.
	}
	if x >= tanhLimit {
		return 1.
	}
	expx := math.Exp(x)
	expmx := math.Exp(-x)
	up := expx - expmx
	down := expx + expmx
	return up / down
}

func (Tanh) P(x float64) float64 {
	return 1 - math.Pow(x, 2)
}
