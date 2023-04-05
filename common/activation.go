package common

type Activation interface {
	F(float64) float64
	//I(float64) float64
	P(float64) float64
}
