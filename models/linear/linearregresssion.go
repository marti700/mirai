package regression

import (
	"github.com/marti700/mirai/estimators"
	"github.com/marti700/veritas/linearalgebra"
)

// Linear model is an interface that wraps the basic behavior of linear models
type LinearModel interface {
	train(target linearalgebra.Matrix, data linearalgebra.Matrix, learningRate float64)
	predict(data linearalgebra.Matrix, hyperParmeters linearalgebra.Matrix) float64
}

type LinearRegression struct {
	Hyperparameters linearalgebra.Matrix
}

// trains a linear model
// the target parameter is the variable to be predicted
// the data parameter is the data observations represented as a matrix
// learningRate is the learning rate at which the model will learn
// this function returns a vector that represents the estimations for the model hyperparameters
func (m LinearRegression) Train(target linearalgebra.Matrix,
	data linearalgebra.Matrix,
	learningRate float64) {

	data, _ = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0)
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

	estimatedHyperParameters := estimators.GradiantDescent(learningRate, &data, &target, gradient)

	m.Hyperparameters = *estimatedHyperParameters
}

//gives a prediction based on the hyperparameters estimations obtained by train()
func (m LinearRegression) predict(data linearalgebra.Matrix, hypparameters linearalgebra.Matrix) float64 {
	return 0
}
