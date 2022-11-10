package utils

import (
	"math/rand"
	"time"

	"github.com/marti700/veritas/linearalgebra"
)

// represents a Resampling Fold that contains test and training data
// The fields:
// Train: is the data used for training the model
// Test: is the data used for testing the model
// TargetTrain is the target variable (for wich we want to predict) of the raining set (The Train property)
// TargetTest is the target variable (for wich we want to predict) of the testing set (The Test property)
type Fold struct {
	Train       linearalgebra.Matrix
	Test        linearalgebra.Matrix
	TargetTrain linearalgebra.Matrix
	TargetTest  linearalgebra.Matrix
}

func Bootstrap(data linearalgebra.Matrix) (linearalgebra.Matrix, linearalgebra.Matrix) {
	dataSize := data.Row
	inCache := make(map[int]int) // to keep records of the selected rows

	var in linearalgebra.Matrix
	rand.Seed(time.Now().UnixNano())

	// randomly selects the rows for the new dataset
	for i := 0; i < dataSize; i++ {
		row := rand.Intn(dataSize - 1)
		inCache[row] = row
		if linearalgebra.IsEmpty(in) {
			in = data.GetRow(row)
		} else {
			in = linearalgebra.Insert(data.GetRow(row), in, in.Row)
		}
	}

	// gets the non selected rows
	var out linearalgebra.Matrix
	for i := 0; i < dataSize; i++ {
		if _, ok := inCache[i]; !ok {
			if linearalgebra.IsEmpty(out) {
				out = data.GetRow(i)
			} else {
				out = linearalgebra.Insert(data.GetRow(i), out, out.Row)
			}
		}
	}

	return in, out
}

func CrossValidate(data, target linearalgebra.Matrix, folds int) []Fold {

	cross_val := make([]Fold, folds)

	folds_diff := data.Row / folds
	currentTestFoldStart := 0
	currentTestFoldEnd := folds_diff

	// for each fold
	for i := 0; i < folds; i++ {
		var test_fold linearalgebra.Matrix
		var train_fold linearalgebra.Matrix

		var target_test_fold linearalgebra.Matrix
		var target_train_fold linearalgebra.Matrix
		// traverse the data matrix
		for k := 0; k < data.Row; k++ {
			// when in range get the test fold
			if k >= currentTestFoldStart && k < currentTestFoldEnd {
				if linearalgebra.IsEmpty(test_fold) {
					test_fold = data.GetRow(k)
					target_test_fold = target.GetRow(k)
				} else {
					test_fold = linearalgebra.Insert(data.GetRow(k), test_fold, test_fold.Row)
					target_test_fold = linearalgebra.Insert(target.GetRow(k), target_test_fold, target_test_fold.Row)
				}
			} else { //get the training fold
				if linearalgebra.IsEmpty(train_fold) {
					train_fold = data.GetRow(k)
					target_train_fold = target.GetRow(k)
				} else {
					train_fold = linearalgebra.Insert(data.GetRow(k), train_fold, train_fold.Row)
					target_train_fold = linearalgebra.Insert(target.GetRow(k), target_train_fold, target_train_fold.Row)
				}

			}
		}

		fold := Fold{
			Test:        test_fold,
			Train:       train_fold,
			TargetTrain: target_train_fold,
			TargetTest:  target_test_fold,
		}
		cross_val[i] = fold

		currentTestFoldStart = currentTestFoldEnd
		currentTestFoldEnd = folds_diff + currentTestFoldStart

	}
	return cross_val
}
