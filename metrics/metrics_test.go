package metrics

import (
	"testing"

	"github.com/marti700/mirai/models/linearmodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
	"github.com/marti700/mirai/utils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestMeanSquareError(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	predicted := linearalgebra.NewColumnVector([]float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	expected := 33.0
	result := MeanSquareError(actual, predicted)

	if result != 33.0 {
		t.Error("Expected result is: ", expected, "but was :", result)
	}

}

func TestAcc(t *testing.T) {
	predicted1 := linearalgebra.NewColumnVector([]float64{0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1})
	predicted2 := linearalgebra.NewColumnVector([]float64{0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0})
	actual := linearalgebra.NewColumnVector([]float64{0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0})

	expected1 := 0.8333333333333334
	expected2 := 0.90

	acc1 := Acc(predicted1, actual)
	acc2 := Acc(predicted2, actual)

	if expected1 != acc1 {
		t.Error("Expected result is: ", expected1, "but was :", acc1)
	}

	if expected2 != acc2 {
		t.Error("Expected result is: ", expected2, "but was :", acc2)
	}

}

func TestRSS(t *testing.T) {
	predicted := linearalgebra.NewColumnVector([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	actual := linearalgebra.NewColumnVector([]float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100})

	expected := 31185.0
	result := RSS(actual, predicted)

	if result != expected {
		t.Error("Expected result is: ", expected, "but actual was :", result)
	}
}

func TestCrossValidate(t *testing.T) {

	// get data
	data := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/X_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/y_train.csv")
	expectedResult := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/cross_val_scores.csv")

	// sklearn cross val score using the mean squared error are negated here we convert the scores to positive values
	expectedResult = expectedResult.Map(func(x float64) float64 {
		return -1.0 * x
	})
	// get cross validation folds
	kFolds := 10
	crossValidation := utils.CrossValidate(data, target, kFolds)

	// train a linear regression model
	options := options.LROptions{
		Estimator: options.NewGDOptions(1000, 0.001, 0.00003),
	}
	lr := &linearmodels.LinearRegression{Opts: options}

	//get the cross validation score
	crossValScores := CrossValidationScore(crossValidation, lr, MeanSquareError)

	if len(crossValScores) != 10 {
		t.Error("The number of folds are expected to be 10")
	}

	if !testutils.AcceptableResults(expectedResult, linearalgebra.NewColumnVector(crossValScores), 50) {
		t.Error("Error expected result is ", expectedResult, " but was", crossValScores)
	}
}
func TestCrossValidateConcurrent(t *testing.T) {

	// get data
	data := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/X_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/y_train.csv")
	expectedResult := testutils.ReadDataFromcsv("../testdata/datagenerators/data/crossvalidation/data/cross_val_scores.csv")

	// sklearn cross val score using the mean squared error are negated here we convert the scores to positive values
	expectedResult = expectedResult.Map(func(x float64) float64 {
		return -1.0 * x
	})
	// get cross validation folds
	kFolds := 10
	crossValidation := utils.CrossValidate(data, target, kFolds)

	// train a linear regression model
	options := options.LROptions{
		Estimator: options.NewGDOptions(1000, 0.001, 0.00003),
	}
	lr := &linearmodels.LinearRegression{Opts: options}

	//get the cross validation score
	crossValScores := CrossValidationScoreConcurrent(crossValidation, lr, MeanSquareError)

	if len(crossValScores) != 10 {
		t.Error("The number of folds are expected to be 10")
	}

	if !testutils.AcceptableResults(expectedResult, linearalgebra.NewColumnVector(crossValScores), 50) {
		t.Error("Error expected result is ", expectedResult, " but was", crossValScores)
	}
}

func TestRSquared(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{3, 8, 10, 17, 24, 27})
	predicted := linearalgebra.NewColumnVector([]float64{2, 8, 10, 13, 18, 20})

	result := RSquared(actual, predicted)

	if result != 0.7717269675494218 {
		t.Error("Error expected result is 0.7717269675494218", " but was", result)

	}
}

func TestGetConfusionMatrix(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})
	predicted := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})

	result := GetConfusionMatrix(actual, predicted)

	expectedResult := make([]ConfusionMatrix, 3)

	expectedResult[0] = ConfusionMatrix{TP: 16, FP: 0, FN: 0, TN: 29}
	expectedResult[1] = ConfusionMatrix{TP: 17, FP: 0, FN: 1, TN: 27}
	expectedResult[2] = ConfusionMatrix{TP: 11, FP: 1, FN: 0, TN: 33}

	for i, r := range result {
		if r != expectedResult[i] {
			t.Error("Error expected result is ", expectedResult[i], " but was", r)
		}
	}
}

func TestGetAccuarcy(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})
	predicted := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})

	result := GetConfusionMatrix(actual, predicted)
	expecterResult := 1.0

	if result[0].GetAccuarcy() != 1.0 {
		t.Error("Error expected result is", expecterResult, "but was", result[0].GetAccuarcy())
	}
}

func TestGetPrecision(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})
	predicted := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})

	result := GetConfusionMatrix(actual, predicted)
	expecterResult := 1.0

	if result[1].GetPrecision() != 1.0 {
		t.Error("Error expected result is", expecterResult, "but was", result[0].GetPrecision())
	}
}

func TestGetRecall(t *testing.T) {
	actual := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})
	predicted := linearalgebra.NewColumnVector([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3})

	result := GetConfusionMatrix(actual, predicted)
	expecterResult := 1.0

	if result[1].GetRecall() != 0.9444444444444444 {
		t.Error("Error expected result is", expecterResult, "but was", result[0].GetRecall())
	}
}
