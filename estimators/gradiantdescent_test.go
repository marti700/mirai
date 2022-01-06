package estimators_test

import (
	"testing"

	"github.com/marti700/mirai/estimators"
	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestGradintDescent(t *testing.T) {
	//POINTS
	// (0.5,1.4)
	// (2.9,3.2)
	// (2.3,1.9)

	// the gradient used is the one of the MSE function assuming that used on linear regression
	// extra 1 is added to the data to falcilitate the calculation of the intercept term of the
	// linear regreassion model
	data := linearalgebra.NewMatrix([][]float64{
		{1, 0.5},
		{1, 2.9},
		{1, 2.3},
	})

	target := linearalgebra.NewMatrix([][]float64{
		{1.4},
		{3.2},
		{1.9},
	})

	// mean square error function gradient
	gradient := func(tv linearalgebra.Matrix, f linearalgebra.Matrix, slopes linearalgebra.Matrix) linearalgebra.Matrix {
		lossFunctionVals := make([]float64, f.Col)

		//model predictios with the currently estimated slope values
		currentModelResults, _ := f.Mult(slopes)
		for j := 0; j < f.Col; j++ {
			predErr, _ := tv.Substract(currentModelResults)
			result, _ := f.GetCol(j).ScaleBy(-2).HadamardProduct(predErr)
			lossFunctionVals[j] = linearalgebra.ElementsSum(result)
		}
		return linearalgebra.NewColumnVector(lossFunctionVals)
	}

	gdr := estimators.GradiantDescent(0.01, &data, &target, gradient)

	expected := linearalgebra.NewColumnVector([]float64{0.95, 0.65})

	if !testutils.AcceptableReslts(expected, *gdr) {
		t.Error("Error expected result is ", expected, " but was", *gdr)
	}
}
