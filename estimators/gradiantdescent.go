package estimators

import (
	"errors"
	"github.com/marti700/veritas/linearalgebra/matrix"
	"github.com/marti700/veritas/linearalgebra/vector"
)

// Execute the gradiant descent algorithm
// learningRate: is the algorithm learning rate
// data: is the matrix on which the hyperparameters will be estimated
// target: the variable we are trying to predict
// gradient: the gradient of the loss function
func GradiantDescent(learningRate float64,
	data *matrix.Matrix,
	target *vector.Vector,
	gradient []func(target []float64,
		data *matrix.Matrix,
		slopes []float64) float64) (*vector.Vector, error) {

	if data.Col != len(gradient) {
		return nil, errors.New("matrix dimesion should be equal to gradient size")
	}

	// generate slopes initial values (I guess they can be zero) -
	slopesVal := make([]float64, len(gradient))
	tempSlopeVals := make([]float64, len(gradient))

	for i := 0; i < 1000; i++ {
		for j := 0; j < len(gradient); j++ {
			tempSlopeVals[j] = tempSlopeVals[j] - (learningRate * gradient[j](target.Data, data, slopesVal))
		}
		// after eatch iteration simultaniuosly update the slopes
		copy(slopesVal, tempSlopeVals)
	}
	newVectorsVal := vector.NewVector(slopesVal)
	return &newVectorsVal, nil
}
