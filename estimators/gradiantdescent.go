package estimators

import (
	"errors"

	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/veritas/commons"
)

// Execute the gradiant descent algorithm
// learningRate: is the algorithm learning rate
// data: is the matrix on which the hyperparameters will be estimated
// target: the variable we are trying to predict
// gradient: the gradient of the loss function
func GradiantDescent(learningRate float64,
	data *linearalgebra.Matrix,
	target *linearalgebra.Matrix,
	gradient func(target *linearalgebra.Matrix,
		data *linearalgebra.Matrix,
		slopes linearalgebra.Matrix) linearalgebra.Matrix) (*linearalgebra.Matrix, error) {

	if data.Col != target.Row {
		return nil, errors.New("data matrix features should be equal to target size")
	}

	// generate slopes initial values (I guess they can be zero) -
	slopesVal := linearalgebra.NewColumnVector(make([]float64, data.Col))
	tempSlopeVals := linearalgebra.NewColumnVector(make([]float64, data.Col))

	for i := 0; i < 1000; i++ {
		for j := 0; j < data.Col; j++ {
			tempSlopeVals.Data[j] = tempSlopeVals.Data[j] - (learningRate * commons.Sum(gradient(target, data, slopesVal).Data))
		}
		// after eatch iteration simultaniuosly update the slopes
		copy(slopesVal.Data, tempSlopeVals.Data)
	}
	newVectorsVal := slopesVal
	return &newVectorsVal, nil
}
