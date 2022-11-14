package utils

import (
	"testing"

	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)

func TestBootstrap(t *testing.T) {

	data := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")

	inBag, outOfBag := Bootstrap(data)

	if linearalgebra.IsEmpty(inBag) || linearalgebra.IsEmpty(outOfBag) {
		t.Error("one of inBag or outOfBag sets are empty")
	}
}

func TestCrossValidate(t *testing.T) {

	data := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")
	target := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/y_train.csv")

	kFolds := 10

	crossValidation := CrossValidate(data, target, kFolds)

	if len(crossValidation) != 10 {
		t.Error("The number of folds are expected to be 10")
	}

	for _, v := range crossValidation {
		testRows := data.Row / kFolds
		trainRows := data.Row - testRows
		if v.Train.Row != trainRows || v.Test.Row != testRows || v.TargetTrain.Row != trainRows || v.TargetTest.Row != testRows {
			t.Error("Train rows must be ", trainRows, "test rows must be", testRows)
		}

		if v.Train.Col != data.Col || v.Test.Col != data.Col || v.TargetTrain.Col != 1 || v.TargetTest.Col != 1 {
			t.Error("Train and Test columns must be equal to the same number of colums of the data (", data.Col, ") target columns must be 1")
		}
	}
}
