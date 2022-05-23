package treemodels

import (
	"sort"

	"github.com/marti700/veritas/linearalgebra"
)

// returns a slice of the midpoints between the sorted element of a vector
// data: is the vector to be proccessed. It must be a column vector
func getMidPoints(data linearalgebra.Matrix) []float64 {
	if !linearalgebra.IsColumnVector(data) {
		panic("input matrix must be a column vector")
	}

	sort.Float64s(data.Data)
	r := append(data.Data, data.Data[len(data.Data)-1]+1) //adds extra element to obtain a midpoint above the last number
	midPoints := make([]float64, 0, len(r))

	i := 0
	j := 1

	for j < len(r) {
		midPoints = append(midPoints, (r[i]+r[j])/2.0)
		i++
		j++
	}

	return midPoints
}

// returns the average of a slice of float64 numbers
func average(data []float64) float64 {
	var sum float64
	for _, elm := range data {
		sum += elm
	}

	return sum / float64(len(data))
}