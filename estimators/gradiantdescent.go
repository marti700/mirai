package estimators

import (
	"errors"
	"github.com/marti700/veritas/linearalgebra/matrix"
	"github.com/marti700/veritas/linearalgebra/vector"
)

func GradiantDescent(s float64, data *matrix.Matrix, target *vector.Vector, gradient []func(...float64)) (*vector.Vector, error) {
	if data.Col != len(gradient) {
		return nil, errors.New("matrix dimesion should be equal to gradient size")
	}

	// transpose the matrix to get its columns
	dataT := data.T()
	// append 1 to each column

	// generate slopes initial values (i guess they can be zero)
	slopesVal := vector.NewVector(make([]float64, len(gradient)))
	tempSlopeVals := make([]float64, len(gradient))

	// multiply the data by the slope values
	// after eatch iteration simultaniuosly update the slopes


	return nil, nil

}