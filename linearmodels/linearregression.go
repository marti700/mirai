package linearmodels

import (
	"github.com/marti700/mirai/estimators"
	"github.com/marti700/mirai/options"
	"github.com/marti700/veritas/linearalgebra"
)

var regularizators = map[string]func(param, lambda float64) float64{}

func init() {
	none := func(param, lambda float64) float64 { return 0 }
	l2 := func(param, lambda float64) float64 { return param * lambda * 2 }
	//using subgradient since the absolute value function derivative is not defined at zero
	// see https://machinelearningcompass.com/machine_learning_math/subgradient_descent/
	l1 := func(param, lambda float64) float64 {return lambda * sign(param)	}

	regularizators[""] = none
	regularizators["l2"] = l2
	regularizators["l1"] = l1
}

// Returns an element-wise indication of the sign of a number.
// returns -1 if the number is negative 0 if the number is 0 and 1 if the number is positive
func sign(x float64) float64 {
	if x < 0 {
		return -1
	} else if x == 0 {
		return 0
	}
	return 1
}

// returns the regularization penalty
func regularizator(param float64, opt options.RegOptions) float64 {
	return regularizators[opt.Type](param, opt.Lambda)
}

// Linear model is an interface that wraps the basic behavior of linear models
type LinearModel interface {
	Train(target linearalgebra.Matrix, data linearalgebra.Matrix, options options.LROptions)
	Predict(data linearalgebra.Matrix) linearalgebra.Matrix
}

// general Linear model linear model struct type
type LinearRegression struct {
	Hyperparameters linearalgebra.Matrix
}

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
		if learningRate == 0.0 {
			panic("Learning rate is 0")
		}

		data = linearalgebra.Insert(linearalgebra.Ones(data.Row, 1), data, 0)
		gradient := func(tv linearalgebra.Matrix, f linearalgebra.Matrix, slopes linearalgebra.Matrix) linearalgebra.Matrix {
			lossFunctionVals := make([]float64, f.Col)

			//model predictios with the currently estimated slope values
			currentModelResults := f.Mult(slopes)
			for j := 0; j < f.Col; j++ {
				predErr := tv.Substract(currentModelResults)
				lossFunctionVals[j] = linearalgebra.ElementsSum(f.GetCol(j).ScaleBy(-2).HadamardProduct(predErr)) + regularizator(slopes.Get(j, 0), opt.Regularization)
			}
			return linearalgebra.NewColumnVector(lossFunctionVals)
		}
		estimatedHyperParameters := estimators.GradiantDescent(learningRate, &data, &target, gradient)
		m.Hyperparameters = *estimatedHyperParameters
	} else if opt.Estimator.GetType() == "ols" {
		data = linearalgebra.Insert(linearalgebra.Ones(data.Row, 1), data, 0)
		m.Hyperparameters = estimators.OLS(data, target)
	}
}

//gives predictions based on the hyperparameters estimations obtained by Train()
func (m LinearRegression) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	data = linearalgebra.Insert(linearalgebra.Ones(data.Row, 1), data, 0) //to take into considaration the bias term
	return data.Mult(m.Hyperparameters)
}
