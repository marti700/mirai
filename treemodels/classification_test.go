package treemodels

import (
	"testing"

	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestGetGinis(t *testing.T) {
	data := linearalgebra.NewMatrix([][]float64 {
		{1,1,4},
		{2,2,3},
		{1,1,4},
		{1,3,3},
		{3,4,10},
	})

	target := linearalgebra.NewColumnVector([]float64{1,2,1,3,4})


	predicted := Predict(data, Train(data,target))
	if !testutils.AcceptableResults(target, predicted, 0) {
		t.Error("Error expected result is ", target, " but was", predicted)
	}

}