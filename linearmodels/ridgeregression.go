package linearmodels

import (
	// "github.com/marti700/mirai/estimators"
	"github.com/marti700/mirai/options"
	"github.com/marti700/veritas/linearalgebra"
)

// trains a linear model with l2 regularization (ridge regression)
// the target parameter is the variable to be predicted
// the data parameter is the data observations represented as a matrix
// this function will set the hyperparameter directly in the reciver and will panic if
// LROptions.Estimator = "gd" and LROptinos.LearningRate = 0

func (m *RidgeRegression) Train(target linearalgebra.Matrix, data linearalgebra.Matrix, opt options.RegOptions) {
	// if opt.Estimator.GetType() == "gd" {
	// 	if opt.Estimator.(options.GDOptions).LearningRate == 0 {
	// 		panic("learning rate is 0")
	// 	}
	// 	data = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0)
	// 	gradient := func(tv linearalgebra.Matrix, f linearalgebra.Matrix, slopes linearalgebra.Matrix) linearalgebra.Matrix {
	// 		lossFunctionVals := make([]float64, f.Col)
	// 		modelPred := f.Mult(slopes)
	// 		for i := 0; i < f.Col; i++ {
	// 			errors := tv.Substract(modelPred)
	// 			lossFunctionVals[i] = linearalgebra.ElementsSum(data.GetCol(i).ScaleBy(-2).HadamardProduct(errors)) + (2 * slopes.Get(i, 0) * opt.Lambda)
	// 		}
	// 		return linearalgebra.NewColumnVector(lossFunctionVals)
	// 	}
	// 	m.Hyperparameters = *estimators.GradiantDescent(opt.Estimator.(options.GDOptions).LearningRate, &data, &target, gradient)
	// }
}

//gives predictions based on the hyperparameters estimations obtained by Train()
func (m *RidgeRegression) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	data = data.InsertAt(linearalgebra.Ones(data.Row, 1), 0) //to take into considaration the bias term
	return data.Mult(m.Hyperparameters)
}
