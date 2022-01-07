package testutils

import (
	"github.com/marti700/veritas/linearalgebra"
	"math"
)

func AcceptableReslts(expected, actual linearalgebra.Matrix) bool {
	for i := range expected.Data {
		absoluteDelta :=  math.Abs(expected.Data[i]-actual.Data[i])
		if absoluteDelta > 2 || math.IsNaN(absoluteDelta)  {
			return false
		}
	}
	return true
}
