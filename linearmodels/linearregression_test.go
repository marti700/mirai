package linearmodels_test

import (
	"testing"

	"github.com/marti700/mirai/linearmodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestLinearRegressionTrain(t *testing.T) {
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

	lr := linearmodels.LinearRegression{}

	options := options.LROptions {
		Estimator: options.NewGDOptions(1000, 0.01, 0.00003),
	}

	lr.Train(target, data, options)

	// expected hyper parameter estimations
	expected := linearalgebra.NewColumnVector([]float64{0.95, 0.65})

	if !testutils.AcceptableReslts(expected, lr.Hyperparameters) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}
}
