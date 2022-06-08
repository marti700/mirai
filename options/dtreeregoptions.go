// this file contains some configuration options for the decision tree regressor

package options

import "github.com/marti700/veritas/linearalgebra"

// Decision tree regressor options
// the criterion parameter controls the criteria used to split the tree.
//	 It must be a two argument fuction which argumets are of type linearalgebra.Matrix
//	 is recomended to use the MeanSquareError and RSS functions from the metrics package
// The MinLeafSamples controls the minimun number samples to consider the data to be a leaf
//
type DTreeRegreessorOptions struct {
	Criterion      func(actual, predicted linearalgebra.Matrix) float64
	MinLeafSamples int
}

func NewDTRegressorOptions(minLeafSamples int,
	criteria func(actual, predicted linearalgebra.Matrix) float64) DTreeRegreessorOptions {

	return DTreeRegreessorOptions{
		Criterion:      criteria,
		MinLeafSamples: minLeafSamples,
	}
}

func (o DTreeRegreessorOptions) GetType() string {
	return "DTREGRESSOR"
}
