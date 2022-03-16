package metrics

import (
	"github.com/marti700/veritas/linearalgebra"
)

// calculates the mean square error of two column vectors
func MeanSquareError (actual, predicted linearalgebra.Matrix) float64 {
	dataPoints := actual.Row
	squareF := func (x float64) float64 {return x*x}
	return linearalgebra.ElementsSum(actual.Substract(predicted).Map(squareF))/float64(dataPoints)
}