package treemodels

import (
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

func BuildTree1(data, target linearalgebra.Matrix) Tree {
	if target.Row < 21 {
		return Tree{
			Left:    nil,
			Right:   nil,
			Predict: stats.Mean(target.Data),
		}
	}


	featsRSS := make([]float64, data.Col)
	midPoint := math.Inf(1)
	// for each feature
	for i := 0; i < data.Col; i++ {
		feat := data.GetCol(i)
		dataTarget := linearalgebra.Insert(target, feat, 2)
		midPoints := getMidPoints(feat)
		// uu := make([]float64, len(midPoints))
		minRSS := math.Inf(1)
		// calculate the min RSS
		for j := 0; j < len(midPoints)-1; j++ {
			less, greater := linearalgebra.Filter2(dataTarget, func(r linearalgebra.Matrix) bool {
				return r.Get(0,0) < midPoints[j]
			}, 0)
			// less, greater := linearalgebra.ElementWiseFilter2(feat, func(r float64) bool {
			// 	return r < midPoints[j]
			// }, 1)

			lessMean := genOneValueVector(
				stats.Mean(less.GetCol(1).Data),
				less.Row,
			)

			greaterMean := genOneValueVector(
				stats.Mean(greater.GetCol(1).Data),
				greater.Row,
			)

			currentRSS := metrics.RSS(less.GetCol(1), lessMean) + metrics.RSS(greater.GetCol(1), greaterMean)
			// uu[j] = currentRSS
			if minRSS > currentRSS {
				minRSS = currentRSS
				midPoint = midPoints[j]
			}
		}
		// uu = make([]float64, len(midPoints))
		featsRSS[i] = minRSS
	}

	minValIdx := min(featsRSS) // the index of the minimun value
	stats.Min(featsRSS)
	tmpDat := linearalgebra.Insert(target, data, data.Col+1)
	left, right := linearalgebra.Filter2(data, func(r linearalgebra.Matrix) bool {
		return tmpDat.Get(0, minValIdx) < featsRSS[minValIdx]
	}, 0)

	return Tree{
		Left:      buildTree(left),
		Right:     buildTree(right),
		feature:   minValIdx,
		Condition: midPoint,
	}
}
