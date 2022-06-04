package treemodels

import (
	// "fmt"
	"math"

	"github.com/marti700/mirai/metrics"
	"github.com/marti700/veritas/linearalgebra"
	"github.com/marti700/veritas/stats"
)

// 1- get feture RSS
// 2- Seelect the feature with the lowest RSS

// Finds the minimun RSS for this feature
// func fRSS(f, target linearalgebra.Matrix) float64 {
// 	midPoints := getMidPoints(f)
// 	var minRSS float64
// 	for i := 0; i < len(midPoints)-1; i++ {
// 		less, greater := linearalgebra.ElementWiseFilter2(f, func(r float64) bool {
// 			return r < midPoints[i]
// 		}, 1)

// 		lessMean := genOneValueVector(
// 			stats.Mean(target.Data[:i+1]),
// 			less.Row,
// 		)

// 		greaterMean := genOneValueVector(
// 			stats.Mean(target.Data[i+1:]),
// 			greater.Row,
// 		)

// 		minRSS = metrics.RSS(lessMean, less) + metrics.RSS(greaterMean, greater)
// 	}

// 	return minRSS
// }

// generates a column vector of length len and which only value will be val
func genOneValueVector(val float64, len int) linearalgebra.Matrix {
	vec := make([]float64, len)
	for i := 0; i < len; i++ {
		vec[i] = val
	}
	return linearalgebra.NewColumnVector(vec)
}

// func selectBestFeature(data, target linearalgebra.Matrix) (int, float64) {
// 	featsRSS := make([]float64, data.Col)
// 	for i := 0; i < data.Col; i++ {
// 		featsRSS[i] = fRSS(data.GetCol(i), target)
// 	}

// 	minValIdx := min(featsRSS) // the index of the minimun value

// 	return minValIdx, featsRSS[minValIdx]
// }

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
		// dataTarget := linearalgebra.Insert(target, feat, 2)
		midPoints := getMidPoints(feat)
		midPoint = math.Inf(1)
		mRSS := make([]float64, len(midPoints))
		// uu := make([]float64, len(midPoints))
		minRSS := math.Inf(1)
		// calculate the min RSS
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
			// uu[j] = currentRSS
			if minRSS > currentRSS {
				minRSS = currentRSS
				midPoint = midPoints[j]
			}
		}
		// uu = make([]float64, len(midPoints))
		featsRSS[i] = minRSS
		featMidpoint[i] = midPoint
		// featsRSS[i] = stats.Mean(mRSS)

	}

	minValIdx := min(featsRSS) // the index of the minimun value
	optimalMidpoint := featMidpoint[minValIdx]
	stats.Min(featsRSS)
	// s,g := linearalgebra.Filter2(data, func(r linearalgebra.Matrix) bool {
	// 	return r.Get(0,6) < -0.592},0)
	// fmt.Println(s)
	// fmt.Println(g)
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
