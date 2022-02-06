package linearmodels_test

import (
	"testing"

	"github.com/marti700/mirai/linearmodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
)

func TestLinearRegressionTrainGD(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/y_train.csv")

	lr := linearmodels.LinearRegression{}

	options := options.LROptions{
		Estimator: options.NewGDOptions(1000, 0.001, 0.00003),
	}

	lr.Train(target, trainData, options)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lr.Hyperparameters, 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/predictions.csv")

	if !testutils.AcceptableResults(expectedPredictions, lr.Predict(testData), 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}
}

func TestLinearRegressionTrainOLS(t *testing.T) {
	trainData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/y_train.csv")

	lr := linearmodels.LinearRegression{}

	options := options.LROptions{
		Estimator: options.NewOLSOptions(),
	}

	lr.Train(target, trainData, options)

	// expected hyper parameter estimations
	expected := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/hyperparameters.csv")

	if !testutils.AcceptableResults(expected, lr.Hyperparameters, 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}

	//predicted test values
	testData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/x_test.csv")
	expectedPredictions := testutils.ReadDataFromcsv("../testdata/datagenerators/data/linearregression/data/predictions.csv")

	if !testutils.AcceptableResults(expectedPredictions, lr.Predict(testData), 0.001) {
		t.Error("Error expected result is ", expected, " but was", lr.Hyperparameters)
	}

}
