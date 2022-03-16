package metrics_test

import (
	"testing"
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/mirai/metrics"
)

func TestMeanSquareError(t *testing.T) {
	actual := linearalgebra.NewColumnVector(   []float64 {1,2,3,4,5,6,7,8,9,10})
	predicted := linearalgebra.NewColumnVector([]float64 {10,9,8,7,6,5,4,3,2,1})
	expected := 33.0
	result :=  metrics.MeanSquareError(actual,predicted)

	if result != 33.0 {
		t.Error("Expected result is: ",expected, "but was :", result )
	}

}