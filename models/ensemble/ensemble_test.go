package ensamble

import (
	"testing"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/mirai/models/treemodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestBaggingRegressor(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/y_train.csv")
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/x_test.csv")

	regressorOptions := options.NewDTRegressorOptions(20, metrics.RSS)
	model := treemodels.NewDecisionTreeRegressor(regressorOptions)
	// model.Train(trainData, target)

	bag := BaggingRegressor{
		Model:    model,
		N_models: 3,
	}

	bag.Train(trainData, target)
	predictions := bag.Predict(testData)

	if linearalgebra.IsEmpty(predictions) {
		t.Error("Predictions can't be empty")
	}

	// The code below is comment because the ensemble models cannot be compared with sklearn ensemble models because the results
	// depends in the ressampling of the data. Resampling is done randomly selecting data from the same datasource. This means
	// that training a bagging model on the same data will yield a different model each time which in turn will give different predictions.

	// actual := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/y_test.csv")
	// sklearnPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/predictions.csv")

	// myModelMSE := metrics.MeanSquareError(actual, predictions)
	// sklearnModelMSE := metrics.MeanSquareError(actual, sklearnPredictions)

	// if !testutils.IsAcceptableAccuarcyDiff(myModelMSE, sklearnModelMSE, 5000)  {
	// 	t.Error("Predictions can't be empty")
	// }
}

func TestBaggingClassifer(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/y_train.csv")

	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_test.csv")

	model := treemodels.NewDecicionTreeeClassifier(options.NewDTreeClassifierOption("ENTROPY"))
	model.Train(trainData, target)

	bag := BaggingClassifier{
		Model:    model,
		N_models: 3,
	}

	bag.Train(trainData, target)
	predictions := bag.Predict(testData)

	// Ensemble models cannot be compared with sklearn ensemble models because the results
	// depends in the ressampling of the data. Resampling is done randomly selecting data from the same datasource. This means
	// that training a bagging model on the same data will yield a different model each time which in turn will give different predictions.

	if linearalgebra.IsEmpty(predictions) {
		t.Error("Predictions can't be empty")
	}

}
