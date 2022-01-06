package testutils

import (
	"github.com/marti700/veritas/linearalgebra"
	"math"
)

func AcceptableReslts(expected, actual linearalgebra.Matrix) bool {
	for i := range expected.Data {
		if math.Abs(expected.Data[i]-actual.Data[i]) > 2 {
			return false
		}
	}
	return true
}
