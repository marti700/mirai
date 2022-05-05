package metrics

import (
	"testing"
	"github.com/marti700/veritas/linearalgebra"
)

func TestMeanSquareError(t *testing.T) {
	actual := linearalgebra.NewColumnVector(   []float64 {1,2,3,4,5,6,7,8,9,10})
	predicted := linearalgebra.NewColumnVector([]float64 {10,9,8,7,6,5,4,3,2,1})
	expected := 33.0
	result :=  MeanSquareError(actual,predicted)

	if result != 33.0 {
		t.Error("Expected result is: ",expected, "but was :", result )
	}

}

func TestAcc(t *testing.T) {
	predicted1 := linearalgebra.NewColumnVector([]float64 {0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,0,0,1})
	predicted2 := linearalgebra.NewColumnVector([]float64 {0,0,1,1,0,1,0,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,0,0,0})
	actual := linearalgebra.NewColumnVector([]float64{0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1,0,1,1,0,1,0})

	expected1 := 0.8333333333333334
	expected2 := 0.90

	acc1 := Acc(predicted1, actual)
	acc2 := Acc(predicted2, actual)

	if expected1 != acc1 {
		t.Error("Expected result is: ",expected1, "but was :", acc1 )
	}

if expected2 != acc2 {
		t.Error("Expected result is: ",expected2, "but was :", acc2 )
	}

}