package model

import (
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/mirai/options"
)

// interface that wraps the basic behavior of models
type Model interface {
	Train(target linearalgebra.Matrix, data linearalgebra.Matrix, options options.LROptions)
	Predict(data linearalgebra.Matrix) linearalgebra.Matrix
}