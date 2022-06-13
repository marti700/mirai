package linearmodels_test

import (
	"testing"

	"github.com/marti700/mirai/models/linearmodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
)

func TestLinearRegressionTrainGD(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/y_train.csv")

	options := options.LROptions{
		Estimator: options.NewGDOptions(1000, 0.001, 0.00003),
	}

	lr := &linearmodels.LinearRegression{Opts: options}
	lr.Train(target, trainData)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lr.Hyperparameters, 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/predictions.csv")
	predicted := lr.Predict(testData)

	if !testutils.AcceptableResults(expectedPredictions, predicted, 50) {
		t.Error("Error expected result is ", expected, " but was", predicted)
	}
}

func TestLinearRegressionTrainOLS(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/y_train.csv")

	options := options.LROptions{
		Estimator: options.NewOLSOptions(),
	}

	lr := &linearmodels.LinearRegression{Opts: options}
	lr.Train(target, trainData)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lr.Hyperparameters, 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/predictions.csv")
	predicted := lr.Predict(testData)

	if !testutils.AcceptableResults(expectedPredictions, predicted, 50) {
		t.Error("Error expected result is ", expected, " but was", predicted)
	}

}

func TestRidgeRegressionTrainGD(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/ridgeregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/ridgeregression/data/y_train.csv")

	options := options.LROptions{
		Estimator:      options.NewGDOptions(1000, 0.001, 0.00003),
		Regularization: options.NewRegOptions("l2", 1.0),
	}

	lm := &linearmodels.LinearRegression{Opts: options}
	lm.Train(target, trainData)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/ridgeregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lm.Hyperparameters, 10) {
		t.Error("Error expected result is ", expected, " but was", lm.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/ridgeregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/ridgeregression/data/predictions.csv")
	predicted := lm.Predict(testData)

	if !testutils.AcceptableResults(expectedPredictions, predicted, 50) {
		t.Error("Error expected result is ", expected, " but was", predicted)
	}
}

func TestLassoRegressionTrainGD(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/lassoregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/lassoregression/data/y_train.csv")


	options := options.LROptions{
		Estimator:      options.NewGDOptions(1000, 0.001, 0.00003),
		Regularization: options.NewRegOptions("l1", 1.0),
	}

	lm := &linearmodels.LinearRegression{Opts: options}
	lm.Train(target, trainData)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/lassoregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lm.Hyperparameters, 10) {
		t.Error("Error expected result is ", expected, " but was", lm.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/lassoregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/lassoregression/data/predictions.csv")
	predicted := lm.Predict(testData)

	if !testutils.AcceptableResults(expectedPredictions, predicted, 50) {
		t.Error("Error expected result is ", expected, " but was", predicted)
	}
}
