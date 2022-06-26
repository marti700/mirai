package model

import (
	"github.com/marti700/veritas/linearalgebra"
)

// interface that wraps the basic behavior of models
type Model interface {
	Train(data, target linearalgebra.Matrix)
	Predict(data linearalgebra.Matrix) linearalgebra.Matrix
	Clone() Model
}