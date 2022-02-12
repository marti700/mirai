package linearmodels

import (
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/mirai/options"
)

// Linear model is an interface that wraps the basic behavior of linear models
type LinearModel interface {
	Train(target linearalgebra.Matrix, data linearalgebra.Matrix, options options.LROptions)
	Predict(data linearalgebra.Matrix) linearalgebra.Matrix
}

// general Linear model linear model struct type
type LinearRegression struct {
	Hyperparameters linearalgebra.Matrix
}

// ridge regression model struct type
type RidgeRegression struct {
	Hyperparameters linearalgebra.Matrix
}
