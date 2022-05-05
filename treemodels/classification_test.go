package treemodels

import (
	"testing"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/mirai/testutils"
)

// Test the classification accuarcy of the descicion tree modeel
func TestTreeAcc(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/y_train.csv")

	actualLabels := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/y_test.csv")
	testData := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/x_test.csv")
	predicted := Predict(testData, Train(trainData, target))
	expectedPredictions := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/predictions.csv")

	myModelAcc := metrics.Acc(predicted, actualLabels )
	skLearnModelAcc := metrics.Acc(expectedPredictions, actualLabels)

	if !testutils.IsAcceptableAccuarcyDiff(myModelAcc, skLearnModelAcc, 0.20) {
		t.Error("Error accuarcy is not acceptable this Model accuarcy is: ", myModelAcc, "but sklearn model accuarcy is :", skLearnModelAcc)
	}

}

