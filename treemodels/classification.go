package treemodels

import (
	"fmt"
	"sort"

	"github.com/marti700/veritas/linearalgebra"
)

// calculates the gini impurity of a feature
// this function recieves the classification classes as a column vector
func giniImpurity(classes linearalgebra.Matrix) float64 {
	classValueCounts := getValueCounts(classes)
	var gini float64

	for _, value := range classValueCounts {
		pValue := float64(value) / float64(classes.Row) // probability of getting this class
		gini += pValue * pValue
	}

	return 1 - gini
}

// returns the index of the feature with less gini impurity (hence the bests spliting feature) and the subfeature with the less impurity
func selectBestSplit(data linearalgebra.Matrix) (int, float64) {
	selectedImp := 42.0
	var bestFeatureIndex int
	var bestMidPoint float64
	for i := 0; i < data.Col-1; i++ {
		currentFeature := data.GetCol(i)
		featureTarget := currentFeature.InsertAt(data.GetCol(data.Col-1), 1)
		midPoints := getMidPoints(currentFeature)
		fImpurities := make([]float64, len(midPoints))

		for j := 0; j < len(midPoints); j++ {
			less := filterRows(featureTarget, func(r linearalgebra.Matrix) bool {
				return r.Get(0, 0) < midPoints[j]
			})

			greater := filterRows(featureTarget, func(r linearalgebra.Matrix) bool {
				return r.Get(0, 0) > midPoints[j]
			})
			fImpurities[j] = (float64(less.Row)/float64(currentFeature.Row))*wrapImp(less) + (float64(greater.Row)/float64(currentFeature.Row))*wrapImp(greater)
		}
		currentFeatureImp := average(fImpurities)

		if selectedImp > currentFeatureImp {
			bestFeatureIndex = i
			selectedImp = currentFeatureImp
			bestMidPoint = midPoints[min(fImpurities)]
		}
	}

	return bestFeatureIndex, bestMidPoint
}

// returns the index of the lowest value of this slice
func min(s []float64) int {
	min := 42.0
	var idx int
	for i, val := range s {
		if min > val {
			min = val
			idx = i
		}
	}
	return idx
}

// some times matrix with no data will be returned and matrix#GetCol will panic with index out of bound when traing to get columns of an emtpy matrix
// this function is a wrapper arround the giniImpurity function so 0 if returned for an empty matrix
func wrapImp(m linearalgebra.Matrix) float64 {
	if m.Row == 0 {
		return 0.0
	}
	return giniImpurity(m.GetCol(1))
}

// returns the average of a slice of float64 numbers
func average(data []float64) float64 {
	var sum float64
	for _, elm := range data {
		sum += elm
	}

	return sum / float64(len(data))
}

func Train(data, target linearalgebra.Matrix) *Tree {
	featureTarget := data.InsertAt(target, data.Col)
	return buildTree(featureTarget)
}

// recursively trains a classification tree and returns the trained tree
func buildTree(data linearalgebra.Matrix) *Tree {
	if len(data.Data) == 0 {
		return &Tree{
			Left:      nil,
			Right:     nil,
			feature:   0,
			Condition: 0,
			Data:      data,
			Predict:   -1,
		}
	}
	target := data.GetCol(data.Col - 1)
	if giniImpurity(target) == 0 {
		return &Tree{
			Left:      nil,
			Right:     nil,
			feature:   0,
			Condition: 0,
			Data:      data,
			Predict:   data.Get(0, data.Col-1),
		}
	}

	// 1- Find best feature split
	bestFeature, bFeatBin := selectBestSplit(data)
	fmt.Println(bestFeature)

	// left is the true branch of the tree and right the false one
	left, right := filterRows2(data, func(r linearalgebra.Matrix) bool {
		return r.Get(0, bestFeature) <= bFeatBin

	})
	// 4- recursively build the tree
	return &Tree{
		Left:      buildTree(left),
		Right:     buildTree(right),
		feature:   bestFeature,
		Condition: bFeatBin,
		Data:      data,
	}
}

func Predict(data linearalgebra.Matrix, t *Tree) linearalgebra.Matrix {
	predictions := make([]float64, data.Row)
	for i := 0; i < data.Row; i++ {
		predictions[i] = classify(data.GetRow(i), t)
	}
	return linearalgebra.NewColumnVector(predictions)
}

// outputs a prediction from a trained tree
func classify(data linearalgebra.Matrix, t *Tree) float64 {
	evFeature := t.feature
	if t.Left == nil && t.Right == nil {
		return t.Predict
	}
	if data.Get(0, evFeature) <= t.Condition {
		return classify(data, t.Left)
	}
	return classify(data, t.Right)
}

// returns a slice of the midpoints between the sorted element of a vector
// data: is the vector to be proccessed. It must be a column vector
func getMidPoints(data linearalgebra.Matrix) []float64 {
	sort.Float64s(data.Data)
	r := append(data.Data, data.Data[len(data.Data)-1]+1) //adds extra element to obtain a midpoint above the last number
	midPoints := make([]float64, 0, len(r))

	i := 0
	j := 1

	for j < len(r) {
		midPoints = append(midPoints, (r[i]+r[j])/2.0)
		i++
		j++
	}

	return midPoints
}

// recieves a column vector as input and returns a map wich keys are the values of the vector
// and its values the number of times the key appears in the vector
func getValueCounts(target linearalgebra.Matrix) map[float64]int {
	if !isVector(target) {
		panic("target must be a column Vector")
	}

	values := make(map[float64]int)

	for i := 0; i < target.Row; i++ {
		currentVal := target.Get(i, 0)
		_, present := values[currentVal]
		if !present {
			values[currentVal] = 1
			continue
		}
		values[currentVal]++

	}
	return values
}

// returns true if a matrix is a column vector false otherwise
func isVector(v linearalgebra.Matrix) bool {
	return  v.Col == 1
}

// given a matrix and a boolean function of the type matrix -> bool returns a new matrix with the elements for what the function returns true
// this function operates on the rows, so each row of the provided matrix will be passed to the boolean function
func filterRows(data linearalgebra.Matrix, f func(r linearalgebra.Matrix) bool) linearalgebra.Matrix {
	var newMatrix linearalgebra.Matrix
	for i := 0; i < data.Row; i++ {
		currentRow := data.GetRow(i)
		if f(currentRow) {
			if len(newMatrix.Data) == 0 {
				newMatrix = currentRow
			} else if f(currentRow) {
				newMatrix = newMatrix.InsertAt(currentRow, 0)
			}
		}
	}
	return newMatrix
}

// given a matrix and a boolean function of the type matrix -> bool returns two new matrices with the elements for what the function returns true
// and the ones for what the function returns false.
//
// this function operates on the rows, so each row of the provided matrix will be passed to the boolean function
func filterRows2(data linearalgebra.Matrix, f func(r linearalgebra.Matrix) bool) (linearalgebra.Matrix, linearalgebra.Matrix) {
	var m1 linearalgebra.Matrix
	var m2 linearalgebra.Matrix
	for i := 0; i < data.Row; i++ {
		currentRow := data.GetRow(i)
		if f(currentRow) && len(m1.Data) == 0 {
			m1 = currentRow
		} else if !f(currentRow) && len(m2.Data) == 0 {
			m2 = currentRow
		} else if f(currentRow) {
			m1 = m1.InsertAt(currentRow, 0)
		} else {
			m2 = m2.InsertAt(currentRow, 0)
		}
	}
	return m1, m2
}
