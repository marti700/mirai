package utils

import (
	"math/rand"

	"github.com/marti700/veritas/linearalgebra"
)

func Bootstrap(data linearalgebra.Matrix) (linearalgebra.Matrix, linearalgebra.Matrix) {
	dataSize := data.Row
	inCache := make(map[int]int) // to keep records of the selected rows

	var in linearalgebra.Matrix

	// randomly selects the rows for the new dataset
	for i := 0; i < dataSize; i++ {
		row := rand.Intn(dataSize - 1)
		inCache[row] = row
		if linearalgebra.IsEmpty(in) {
			in = data.GetRow(i)
		} else {
			in = linearalgebra.Insert(data.GetRow(i), in, in.Row)
		}
	}

	// gets the non selected rows
	var out linearalgebra.Matrix
	for i := 0; i < dataSize; i++ {
		if _, ok := inCache[i]; !ok {
			if linearalgebra.IsEmpty(out) {
				out = data.GetRow(i)
			} else {
				out = linearalgebra.Insert(data.GetRow(i), out, out.Row)
			}
		}
	}

	return in, out
}
