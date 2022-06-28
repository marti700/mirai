package ensamble

import (
	"fmt"
	"testing"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/mirai/models/treemodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
)

// Test the classification accuarcy of the decision tree modeel
func TestClassificationTreeAcc(t *testing.T) {

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

	actual := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/y_test.csv")
	predictions := bag.Predict(testData)
	sklearnPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/baggingregressor/data/predictions.csv")

	myModelMSE := metrics.MeanSquareError(actual, predictions)
	sklearnModelMSE := metrics.MeanSquareError(actual, sklearnPredictions)

	fmt.Println(myModelMSE - sklearnModelMSE)
	if !testutils.IsAcceptableAccuarcyDiff(myModelMSE, sklearnModelMSE, 5000)  {
		t.Error("Predictions can't be empty")
	}

}
