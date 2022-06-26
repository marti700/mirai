package ensamble

import (
	"testing"

	"github.com/marti700/mirai/models/linearmodels"
	"github.com/marti700/mirai/options"
	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

// Test the classification accuarcy of the decision tree modeel
func TestClassificationTreeAcc(t *testing.T) {

	trainData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/y_train.csv")
	testData := testutils.ReadDataFromcsv("../../testdata/datagenerators/data/linearregression/data/x_test.csv")

	options := options.LROptions{
		Estimator: options.NewGDOptions(1000, 0.001, 0.00003),
	}

	bag := BaggingClassifier{
		Model:    &linearmodels.LinearRegression{Opts: options},
		N_models: 3,
	}
	bag.Train(trainData, target)
	predictions := bag.Predict(testData)

	if linearalgebra.IsEmpty(predictions) {
		t.Error("Predictions can't be empty")
	}

}
