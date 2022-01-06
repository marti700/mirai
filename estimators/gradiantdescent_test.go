package estimators_test

import (
	"fmt"
	"testing"

	// "github.com/marti700/mirai/estimators"
	"github.com/marti700/mirai/models/linear"
	"github.com/marti700/veritas/linearalgebra"
)

func TestGradintDescent(t *testing.T) {
	//POINTS
	// (0.5,1.4)
	// (3.2, 2.9)
	// (2.4,1.9)
	data := linearalgebra.NewMatrix([][]float64{
		{0.5},
		{2.9},
		{2.3},
	})

	target := linearalgebra.NewMatrix([][]float64{
		{1.4},
		{3.2},
		{1.9},
	})


	m := regression.LinearRegression{
		Hyperparameters: linearalgebra.NewColumnVector([]float64{0,0}),
	}

	r := m.Train(target, data, 0.01)

	fmt.Println(r)
	// estimators.GradiantDescent(1,nil,nil,nil)
}