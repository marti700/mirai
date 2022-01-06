package estimators

import (
	"github.com/marti700/veritas/linearalgebra"
)

// Execute the gradiant descent algorithm
// learningRate: is the algorithm learning rate
// data: is the matrix on which the hyperparameters will be estimated
// target: the variable we are trying to predict
// gradient: the gradient of the loss function
func GradiantDescent(learningRate float64,
	data *linearalgebra.Matrix,
	target *linearalgebra.Matrix,
	gradient func(target linearalgebra.Matrix,
		data linearalgebra.Matrix,
		slopes linearalgebra.Matrix) linearalgebra.Matrix) *linearalgebra.Matrix {

	// generate slopes initial values (I guess they can be zero ...) -
	slopesVal := linearalgebra.NewColumnVector(make([]float64, data.Col))
	tempSlopeVals := linearalgebra.NewColumnVector(make([]float64, data.Col))

	// TODO: Implement gradiant descent huristics
	// Max number of iteration before giving up
	// Stop when the gradiant stop making significant advances towards the bottom of the hill
	for i := 0; i < 1000; i++ {
		// to determine the slopes values we do:
		// 1- Solves for the slop using the target and data parameters
		// 2- multiplies for the learning rate
		// 3- substract to old step size from the result obtained in the previous operationc
		tempSlopeVals, _ = tempSlopeVals.Substract(gradient(*target, *data, slopesVal).ScaleBy(learningRate))
		// simultaniuosly update the slopes to their new values
		copy(slopesVal.Data, tempSlopeVals.Data)
	}
	newVectorsVal := slopesVal
	return &newVectorsVal
}
