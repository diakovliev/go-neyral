package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func sigma(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func invSigma(x float64) float64 {
	return math.Log(-(x / (x - 1)))
}

// see: https://math.stackexchange.com/questions/1335693/invertible-matrix-of-non-square-matrix
func matRightInversion(A *mat.Dense) (ret *mat.Dense) {

	T := A.T()

	AT := &mat.Dense{}
	AT.Mul(A, T)

	IAT := &mat.Dense{}
	IAT.Inverse(AT)

	R := &mat.Dense{}
	R.Mul(T, IAT)

	ret = &mat.Dense{}
	ret.CloneFrom(R)

	return
}

// see: https://math.stackexchange.com/questions/1335693/invertible-matrix-of-non-square-matrix
func matLeftInversion(A *mat.Dense) (ret *mat.Dense) {

	T := A.T()

	TA := &mat.Dense{}
	TA.Mul(T, A)

	ITA := &mat.Dense{}
	ITA.Inverse(TA)

	R := &mat.Dense{}
	R.Mul(ITA, T)

	ret = &mat.Dense{}
	ret.CloneFrom(R)

	return
}

func main() {

	inputs := 2
	outputs := 3

	data := make([]float64, inputs*outputs)
	for i := range data {
		data[i] = rand.NormFloat64()
	}

	W := mat.NewDense(outputs, inputs, data)

	fmt.Printf("%s\n%v\n", "W = ", mat.Formatted(W, mat.Squeeze()))

	X := mat.NewDense(inputs, 1, []float64{0, 1})

	fmt.Printf("%s\n%v\n", "X = ", mat.Formatted(X, mat.Squeeze()))

	// S = X . W
	S := &mat.Dense{}
	S.Mul(W, X)

	fmt.Printf("%s\n%v\n", "S = ", mat.Formatted(S, mat.Squeeze()))

	// Y = sigma(S)
	Y := &mat.Dense{}
	Y.Apply(func(i, j int, v float64) float64 {
		return sigma(v)
	}, S)

	fmt.Printf("%s\n%v\n", "Y = ", mat.Formatted(Y, mat.Squeeze()))

	// Expectations
	//E := Y.ColView(0)
	E := mat.NewDense(outputs, 1, []float64{0.2, 0.5, 0.8})

	fmt.Printf("%s\n%v\n", "E = ", mat.Formatted(E, mat.Squeeze()))

	T := &mat.Dense{}
	T.Apply(func(i, j int, v float64) float64 {
		return invSigma(v)
	}, E)

	fmt.Printf("%s\n%v\n", "T = ", mat.Formatted(T, mat.Squeeze()))

	// Backward pass
	IW := matLeftInversion(W)
	//IW := matRightInversion(W)
	fmt.Printf("%s\n%v\n", "IW = ", mat.Formatted(IW, mat.Squeeze()))

	OX := &mat.Dense{}
	OX.Mul(IW, T)

	fmt.Printf("%s\n%v\n", "OX = ", mat.Formatted(OX, mat.Squeeze()))
}
