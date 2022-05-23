package metrics

import (
	"github.com/marti700/veritas/linearalgebra"
)

// calculates the mean square error of two column vectors
func MeanSquareError(actual, predicted linearalgebra.Matrix) float64 {
	dataPoints := actual.Row
	squareF := func(x float64) float64 { return x * x }
	return linearalgebra.ElementsSum(actual.Substract(predicted).Map(squareF)) / float64(dataPoints)
}

// calculate the accuarcy of classification
// the predicted argument is the data used for testing the model, the one we want to classify
// the actual argument are the real classes of the dataset
// both predicted and actual must be column vectors of the same dimensions, if they are not this function will panic
func Acc(predicted, actual linearalgebra.Matrix) float64 {
	// TODO: Panic if predicted or actual are not vectors
	// TODO: Panic if predicted and actual are not of the same size
	var diff int
	tElements := len(predicted.Data)
	for i := range predicted.Data {
		if predicted.Data[i] != actual.Data[i] {
			diff++
		}
	}

	return (float64(tElements - diff)) / float64(tElements)
}

// Cualculates the residual squared error of a vector
func RSS(actual, predicted linearalgebra.Matrix) float64 {
	squareF := func(x float64) float64 { return x * x }
	return linearalgebra.ElementsSum(actual.Substract(predicted).Map(squareF))
}