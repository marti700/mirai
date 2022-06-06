package treemodels

import (
	"math"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/veritas/stats"
)


// generates a column vector of length len and which only value will be val
func genOneValueVector(val float64, len int) linearalgebra.Matrix {
	vec := make([]float64, len)
	for i := 0; i < len; i++ {
		vec[i] = val
	}
	return linearalgebra.NewColumnVector(vec)
}

func train1(data, target linearalgebra.Matrix) *Tree {
	dataTarget := linearalgebra.Insert(target, data, data.Col+1)
	return buildTree1(dataTarget)

}

func buildTree1(data linearalgebra.Matrix) *Tree {
	target := data.GetCol(data.Col - 1)
	if target.Row < 20 {
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
		// uu := make([]float64, len(midPoints))
		minRSS := math.Inf(1)
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

			lessRSS = metrics.RSS(less.GetCol(data.Col-1), lessMean)

			greaterMean := genOneValueVector(
				stats.Mean(greater.GetCol(i).Data),
				greater.Row,
			)
			greaterRSS = metrics.RSS(greater.GetCol(data.Col-1), greaterMean)

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
	stats.Min(featsRSS)
	left, right := linearalgebra.Filter2(data, func(r linearalgebra.Matrix) bool {
		return r.Get(0, minValIdx) < optimalMidpoint
	}, 0)

	return &Tree{
		Left:      buildTree1(left),
		Right:     buildTree1(right),
		feature:   minValIdx,
		Condition: optimalMidpoint,
	}
}

func Predict1(data linearalgebra.Matrix, t *Tree) linearalgebra.Matrix {
	predictions := make([]float64, data.Row)
	for i := 0; i < data.Row; i++ {
		predictions[i] = classify(data.GetRow(i), t)
	}
	return linearalgebra.NewColumnVector(predictions)
}
