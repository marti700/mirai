package treemodels

import (
	"math"

	model "github.com/marti700/mirai/models"
	"github.com/marti700/mirai/options"
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/veritas/stats"
)

type DecisionTreeRegressor struct {
	Model *Tree
	Opts options.DTreeRegreessorOptions
}

func NewDecisionTreeRegressor(opt options.DTreeRegreessorOptions) *DecisionTreeRegressor {
	return &DecisionTreeRegressor {
		Opts: opt,
	}
}


// generates a column vector of length len and which only value will be val
func genOneValueVector(val float64, len int) linearalgebra.Matrix {
	vec := make([]float64, len)
	for i := 0; i < len; i++ {
		vec[i] = val
	}
	return linearalgebra.NewColumnVector(vec)
}

// Trains the decision tree regressor based on the data and the target using
// the CART algorithm
func (t *DecisionTreeRegressor) Train(data, target linearalgebra.Matrix) {
	dataTarget := linearalgebra.Insert(target, data, data.Col+1)
	tree := buildRegressionTree(dataTarget, t.Opts)
	t.Model = tree

}

// Recursively builds thee decision tree model based on the data
func buildRegressionTree(data linearalgebra.Matrix, opts options.DTreeRegreessorOptions) *Tree {
	target := data.GetCol(data.Col - 1)
	// if the number of elements in data is less that 20 make a prediction
	// one can go all the way down to 2 or maybe one but this will cause overfit
	if target.Row < opts.MinLeafSamples  {
		return &Tree{
			Left:    nil,
			Right:   nil,
			Predict: stats.Mean(target.Data),
		}
	}

	featsRSS := make([]float64, data.Col-1)
	featMidpoint := make([]float64, data.Col-1)
	var midPoint float64
	// for each feature
	for i := 0; i < data.Col-1; i++ {
		feat := data.GetCol(i)
		midPoints := getMidPoints(feat)
		midPoint = math.Inf(1)
		mRSS := make([]float64, len(midPoints))
		minRSS := math.Inf(1)
		// find the optimal split value
		for j := 0; j < len(midPoints)-1; j++ {
			less, greater := linearalgebra.Filter2(data, func(r linearalgebra.Matrix) bool {
				return r.Get(0, i) < midPoints[j]
			}, 0)

			var lessRSS float64
			var greaterRSS float64

			lessMean := genOneValueVector(
				stats.Mean(less.GetCol(i).Data),
				less.Row,
			)

			lessRSS = opts.Criterion(less.GetCol(data.Col-1), lessMean)

			greaterMean := genOneValueVector(
				stats.Mean(greater.GetCol(i).Data),
				greater.Row,
			)
			greaterRSS = opts.Criterion(greater.GetCol(data.Col-1), greaterMean)

			currentRSS := lessRSS + greaterRSS
			mRSS[j] = currentRSS
			if minRSS > currentRSS {
				minRSS = currentRSS
				midPoint = midPoints[j]
			}
		}
		featsRSS[i] = minRSS
		featMidpoint[i] = midPoint

	}

	minValIdx := min(featsRSS) // the index of the minimun value
	optimalMidpoint := featMidpoint[minValIdx]
	left, right := linearalgebra.Filter2(data, func(r linearalgebra.Matrix) bool {
		return r.Get(0, minValIdx) < optimalMidpoint
	}, 0)

	return &Tree{
		Left:      buildRegressionTree(left, opts),
		Right:     buildRegressionTree(right, opts),
		feature:   minValIdx,
		Condition: optimalMidpoint,
	}
}

// make predictions based on data
// the data argument is a Matrix with similar to the one used for training
// Returns a Matrix containing the predictions for the provided data
func (t *DecisionTreeRegressor) Predict(data linearalgebra.Matrix) linearalgebra.Matrix {
	return genPredictions(data, t.Model)
}

func (t *DecisionTreeRegressor) Clone() model.Model {
	new_model := *t
	return &new_model
}
