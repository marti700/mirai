package linearmodels

import (
	"github.com/marti700/mirai/estimators"
	"github.com/marti700/mirai/options"
	"github.com/marti700/veritas/linearalgebra"
)

// trains a linear model
// the target parameter is the variable to be predicted
// the data parameter is the data observations represented as a matrix
// this function will set the hyperparameter directly in the reciver and will panic if
// LROptions.Estimator = "gd" and LROptinos.LearningRate = 0
func (m *LinearRegression) Train(target linearalgebra.Matrix,
	data linearalgebra.Matrix,
	opt options.LROptions) {

	if opt.Estimator.GetType() == "gd" {
		learningRate := opt.Estimator.(options.GDOptions).LearningRate
		if  learningRate == 0.0 {
			panic("Learning rate is 0")
		}

		data = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0)
		gradient := func(tv linearalgebra.Matrix, f linearalgebra.Matrix, slopes linearalgebra.Matrix) linearalgebra.Matrix {
			lossFunctionVals := make([]float64, f.Col)

			//model predictios with the currently estimated slope values
			currentModelResults := f.Mult(slopes)
			for j := 0; j < f.Col; j++ {
				predErr := tv.Substract(currentModelResults)
				lossFunctionVals[j] = linearalgebra.ElementsSum(f.GetCol(j).ScaleBy(-2).HadamardProduct(predErr))
			}
			return linearalgebra.NewColumnVector(lossFunctionVals)
		}
		estimatedHyperParameters := estimators.GradiantDescent(learningRate, &data, &target, gradient)
		m.Hyperparameters = *estimatedHyperParameters
	} else if opt.Estimator.GetType() == "ols" {
		data = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0)
		m.Hyperparameters = estimators.OLS(data, target)
	}
}

//gives predictions based on the hyperparameters estimations obtained by Train()
func (m LinearRegression) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	data = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0) //to take into considaration the bias term
	return data.Mult(m.Hyperparameters)
}
