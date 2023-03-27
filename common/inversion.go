package common

import "gonum.org/v1/gonum/mat"

// see: https://math.stackexchange.com/questions/1335693/invertible-matrix-of-non-square-matrix
func MatRightInversion(A *mat.Dense) (ret *mat.Dense) {

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
func MatLeftInversion(A *mat.Dense) (ret *mat.Dense) {

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
