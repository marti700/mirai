package treemodels

import (
	"testing"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
)


// Test the classification accuarcy of the decision tree modeel
func TestClassificationTreeAcc(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/y_train.csv")

	actualLabels := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/y_test.csv")
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_test.csv")
	model := NewDecicionTreeeClassifier(options.NewDTreeClassifierOption("GINI"))
	model.Train(trainData, target)
	predicted := model.Predict(testData)
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/predictions.csv")

	myModelAcc := metrics.Acc(predicted, actualLabels )
	skLearnModelAcc := metrics.Acc(expectedPredictions, actualLabels)

	if !testutils.IsAcceptableAccuarcyDiff(myModelAcc, skLearnModelAcc, 0.20) {
		t.Error("Error accuarcy is not acceptable. This Model accuarcy is: ", myModelAcc, "but sklearn model accuarcy is :", skLearnModelAcc)
	}
}

// Test the classification accuarcy of the decision tree modeel
func TestClassificationTreeAccEntropy(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/y_train.csv")

	actualLabels := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/y_test.csv")
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/x_test.csv")
	model := NewDecicionTreeeClassifier(options.NewDTreeClassifierOption("ENTROPY"))
	model.Train(trainData, target)
	predicted := model.Predict(testData)
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/cdecisiontree/data/predictions.csv")

	myModelAcc := metrics.Acc(predicted, actualLabels )
	skLearnModelAcc := metrics.Acc(expectedPredictions, actualLabels)

	if !testutils.IsAcceptableAccuarcyDiff(myModelAcc, skLearnModelAcc, 0.20) {
		t.Error("Error accuarcy is not acceptable. This Model accuarcy is: ", myModelAcc, "but sklearn model accuarcy is :", skLearnModelAcc)
	}
}

func TestRegressionTree(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/rdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/rdecisiontree/data/y_train.csv")

	actualLabels := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/rdecisiontree/data/y_test.csv")
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/rdecisiontree/data/x_test.csv")
	regressorOptions := options.NewDTRegressorOptions(20, metrics.RSS)
	model := NewDecisionTreeRegressor(regressorOptions)
	model.Train(trainData, target)
	predicted := model.Predict(testData)
	// predicted.Plot()
	expectedPredictions := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/rdecisiontree/data/predictions.csv")
	// fmt.Println(predicted)
	// fmt.Println(metrics.MeanSquareError(trainData.GetCol(6), target))

	myModelMSE := metrics.MeanSquareError(actualLabels, predicted)
	skLearnModelMSE := metrics.MeanSquareError(expectedPredictions, actualLabels)

	if !testutils.IsAcceptableAccuarcyDiff(myModelMSE, skLearnModelMSE, 80) {
		t.Error("Error accuarcy is not acceptable. This Model accuarcy is: ", myModelMSE, "but sklearn model accuarcy is :", skLearnModelMSE)
	}
}
