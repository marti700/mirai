package metrics

import (
	model "github.com/marti700/mirai/models"
	"github.com/marti700/mirai/utils"
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/veritas/stats"
)

// Represents a confusion matrix
// TP are the true positive values
// FP are the false positive values
// FN are the false negative values
// TN are the true negative values
type ConfusionMatrix struct {
	TP float64
	FP float64
	FN float64
	TN float64
}

// calculates the accuarcy of the model from a ConfusionMatrix
func (cm ConfusionMatrix) GetAccuarcy() float64 {
	return (cm.TP + cm.TN) / (cm.TP + cm.FP + cm.FN + cm.TN)
}

// calculates the precision of the model from a ConfusionMatrix
func (cm ConfusionMatrix) GetPrecision() float64 {
	return cm.TP / (cm.TP + cm.FP)
}

// calculates the recall of the model from a ConfusionMatrix
func (cm ConfusionMatrix) GetRecall() float64 {
	return cm.TP / (cm.TP + cm.FN)
}

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

// Cualculates the residual sum of squares
func RSS(actual, predicted linearalgebra.Matrix) float64 {
	squareF := func(x float64) float64 { return x * x }
	return linearalgebra.ElementsSum(actual.Substract(predicted).Map(squareF))
}

// calculate the cross validation score of a given model with the given metric function
// given a cross validation fold (that can be obtained with utils.CrossValidation)
// this funciton uses a single thread to produce the scores in the order specified in the folds parameter
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
// given a cross validation fold (that can be obtained with utils.CrossValidation)
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

// Computes R squared
func RSquared(actual, predicted linearalgebra.Matrix) float64 {
	// functions to square a number
	squareF := func(x float64) float64 { return x * x }
	// gets the mean of the actual values of y
	actualMean := stats.Mean(actual.Data)
	//Makes a vector which all its entries are the value for the actualMean)
	actualMeanVector := linearalgebra.Ones(actual.Row, actual.Col).ScaleBy(actualMean)
	// calculate the total sum of squares
	SST := linearalgebra.ElementsSum(actual.Substract(actualMeanVector).Map(squareF))
	RSS := RSS(actual, predicted)

	return 1 - (RSS / SST)
}

// Build a confusion matrices for each classification class
func BuildConfusionMatrices(actual, predicted linearalgebra.Matrix) map[float64]ConfusionMatrix {

	uniqueClassValues := getUniqueValues(actual)
	// cms := make([]ConfusionMatrix, len(uniqueClassValues))
	AllConfMatrices := make(map[float64]ConfusionMatrix)
	for _, cls := range uniqueClassValues {
		AllConfMatrices[cls] = BuildConfusionMatrixFor(cls, actual, predicted)
	}

	return AllConfMatrices
}

func BuildConfusionMatrixFor(class float64, actual, predicted linearalgebra.Matrix) ConfusionMatrix {
	getTPFor := func(class float64, actual, predicted linearalgebra.Matrix) float64 {
		TP := 0.0
		for i, a := range actual.Data {
			if predicted.Data[i] == class && a == class {
				TP += 1
			}
		}
		return TP
	}

	getFPFor := func(class float64, actual, predicted linearalgebra.Matrix) float64 {
		FP := 0.0

		for i, a := range actual.Data {
			if predicted.Data[i] == class && a != class {
				FP += 1
			}
		}

		return FP
	}

	getFNFor := func(class float64, actual, predicted linearalgebra.Matrix) float64 {
		FN := 0.0

		for i, a := range actual.Data {
			if predicted.Data[i] != class && a == class {
				FN += 1
			}
		}

		return FN
	}

	getTNFor := func(class float64, actual, predicted linearalgebra.Matrix) float64 {
		TN := 0.0

		for i, a := range actual.Data {
			if predicted.Data[i] != class && a != class {
				TN += 1
			}
		}
		return TN
	}

	return ConfusionMatrix{
		TP: getTPFor(class, actual, predicted),
		FP: getFPFor(class, actual, predicted),
		FN: getFNFor(class, actual, predicted),
		TN: getTNFor(class, actual, predicted),
	}
}

// recieves a column vector as input and returns a map which keys are the values of the vector
// and its values the number of times the key appears in the vector
func getUniqueValues(target linearalgebra.Matrix) []float64 {
	if !linearalgebra.IsColumnVector(target) {
		panic("target must be a column Vector")
	}

	values := make(map[float64]int)
	uniqueValues := make([]float64, 0, target.Row)

	for i := 0; i < target.Row; i++ {
		currentVal := target.Get(i, 0)
		_, present := values[currentVal]
		if !present {
			values[currentVal] = 1
			uniqueValues = append(uniqueValues, currentVal)
			// continue
		}
	}

	return uniqueValues
}
