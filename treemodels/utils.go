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
