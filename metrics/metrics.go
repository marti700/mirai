package metrics

import (
	model "github.com/marti700/mirai/models"
	"github.com/marti700/mirai/utils"
	"github.com/marti700/veritas/linearalgebra"
)

// calculates the mean square error of two column vectors
func MeanSquareError(actual, predicted linearalgebra.Matrix) float64 {
	dataPoints := actual.Row
	return RSS(actual, predicted) / float64(dataPoints)
}

// calculate the accuarcy of classification
// the predicted argument is the data used for testing the model, the one we want to classify
// the actual argument are the real classes of the dataset
// both predicted and actual must be column vectors of the same dimensions, if they are not this function will panic
func Acc(predicted, actual linearalgebra.Matrix) float64 {
	// TODO: Panic if predicted or actual are not vectors
	// TODO: Panic if predicted and actual are not of the same size
	var diff int
	tElements := len(predicted.Data)
	for i := range predicted.Data {
		if predicted.Data[i] != actual.Data[i] {
			diff++
		}
	}

	return (float64(tElements - diff)) / float64(tElements)
}

// Cualculates the residual squared error of a vector
func RSS(actual, predicted linearalgebra.Matrix) float64 {
	squareF := func(x float64) float64 { return x * x }
	return linearalgebra.ElementsSum(actual.Substract(predicted).Map(squareF))
}

// calculate the cross validation score of a given model with the given metric function
// and given a cross validation fold (that can be obtained with utils.CrossValidation)
// this funciton uses a single thread to produce the scores in the order specified in the Folds parameter
// useful when is necesariy to know what scores belongs to what fold
func CrossValidationScore(folds []utils.Fold,
	model model.Model,
	metric func(linearalgebra.Matrix, linearalgebra.Matrix) float64) []float64 {

	results := make([]float64, len(folds))
	for i, f := range folds {
		model.Train(f.Train, f.TargetTrain)
		results[i] = metric(f.TargetTest, model.Predict(f.Test))
	}
	return results
}

// calculate the cross validation score of a given model with the given metric function concurrently
// and given a cross validation fold (that can be obtained with utils.CrossValidation)
// this funciton uses multiple threads to produce the scores but the orders on which the scores ends in the
// returned slice is not warrantied.
func CrossValidationScoreConcurrent(folds []utils.Fold,
	mod model.Model,
	metric func(linearalgebra.Matrix, linearalgebra.Matrix) float64) []float64 {

	scoreCh := make(chan float64, len(folds))
	defer close(scoreCh)

	score := func(f utils.Fold,
		m model.Model,
		metric func(linearalgebra.Matrix, linearalgebra.Matrix) float64, ch chan float64) {

		mod.Train(f.Train, f.TargetTrain)
		ch <- metric(f.TargetTest, mod.Predict(f.Test))
	}

	for _, f := range folds {
		go score(f, mod, metric, scoreCh)
	}

	results := make([]float64, len(folds))
	for i := 0; i < len(folds); i++ {
		r := <-scoreCh
		results[i] = r
	}

	return results
}
