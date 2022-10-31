package utils

import (
	"math/rand"
	"time"

	model "github.com/marti700/mirai/models"
	"github.com/marti700/veritas/linearalgebra"
)

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

func cross_validate(data, target linearalgebra.Matrix,
	k int,
	t float32,
	model model.Model,
	metric func(linearalgebra.Matrix, linearalgebra.Matrix) float64) {

	cross_val_scores := make([]float64, k)

	folds_diff := int(float32(data.Row) * t)
	currentTestFoldStart := 0
	currentTestFoldEnd := folds_diff

	var test_fold linearalgebra.Matrix
	var train_fold linearalgebra.Matrix

	var target_test_fold linearalgebra.Matrix
	var target_train_fold linearalgebra.Matrix

	// for each fold
	for i := 0; i < k; i++ {
		// traverse the data matrix
		for k := 0; k < data.Row; k++ {
			// when in range get the test fold
			if k >= currentTestFoldStart && k <= currentTestFoldEnd {
				if linearalgebra.IsEmpty(test_fold) {
					test_fold = data.GetRow(k)
					target_test_fold = target.GetRow(k)
				} else {
					linearalgebra.Insert(data.GetRow(k), test_fold, test_fold.Row)
					linearalgebra.Insert(target.GetRow(k), target_test_fold, target_test_fold.Row)
				}
			} else { //get the training fold
				if linearalgebra.IsEmpty(train_fold) {
					train_fold = data.GetRow(k)
					target_train_fold = target.GetRow(k)
				} else {
					linearalgebra.Insert(data.GetRow(k), train_fold, train_fold.Row)
					linearalgebra.Insert(target.GetRow(k), target_train_fold, target_test_fold.Row)
				}

			}

			// get model score for the current fold
			model.Train(data, target)
			cross_val_scores = append(cross_val_scores, metric(target_test_fold, model.Predict(target_train_fold)))

			currentTestFoldEnd = folds_diff + currentTestFoldEnd
			currentTestFoldStart = currentTestFoldEnd + 1
		}
	}
}
