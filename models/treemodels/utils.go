package treemodels

import (
	"math"
	"sort"

	"github.com/marti700/veritas/linearalgebra"
)

// returns a slice of the midpoints between the sorted element of a vector
// data: is the vector to be proccessed. It must be a column vector
func getMidPoints(data linearalgebra.Matrix) []float64 {
	if !linearalgebra.IsColumnVector(data) {
		panic("input matrix must be a column vector")
	}

	tracker := make(map[float64]float64)

	sort.Float64s(data.Data)
	r := append(data.Data, data.Data[len(data.Data)-1]+1) //adds extra element to obtain a midpoint above the last number
	midPoints := make([]float64, 0, len(r))

	i := 0
	j := 1

	for j < len(r) {
		// just procces the data when the contiguous numbers are different
		if r[i] != r[j] {
			candidate := (r[i] + r[j]) / 2.0
			_, present := tracker[candidate]
			// don't insert duplicate point in the final result
			if !present {
				tracker[candidate] = candidate
				midPoints = append(midPoints, candidate)
			}
		}
		i++
		j++
	}

	return midPoints
}

// returns the index of the lowest value of this slice
func min(s []float64) int {
	min := math.Inf(1)
	var idx int
	for i, val := range s {
		if min > val {
			min = val
			idx = i
		}
	}
	return idx
}

// make predictions based on data
// the data argument is a Matrix similar to the one used for training
// Returns a Matrix containing predictions for the provided data
func genPredictions(data linearalgebra.Matrix, t *Tree) linearalgebra.Matrix {
	predictions := make([]float64, data.Row)
	for i := 0; i < data.Row; i++ {
		predictions[i] = makePrediction(data.GetRow(i), t)
	}
	return linearalgebra.NewColumnVector(predictions)
}

// outputs a prediction from a trained tree
func makePrediction(data linearalgebra.Matrix, t *Tree) float64 {
	evFeature := t.feature
	if t.Left == nil && t.Right == nil {
		return t.Predict
	}
	if data.Get(0, evFeature) <= t.Condition {
		return makePrediction(data, t.Left)
	}
	return makePrediction(data, t.Right)
}
