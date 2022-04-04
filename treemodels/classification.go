package treemodels

// 1- find mid points
// 2- filter data (based on midpoints)
// 3- get class probability by features (using filtered data in step 1 as input)
// 2- split de data

import (
	"fmt"
	"sort"

	"github.com/marti700/veritas/linearalgebra"
)

func extractBins(data linearalgebra.Matrix, binsKeys []float64) map[float64]linearalgebra.Matrix {
	bins := make(map[float64]linearalgebra.Matrix)

	//first bin
	bins[binsKeys[0]] = filterRows(data, func(e linearalgebra.Matrix) bool {
		return e.Get(0, 0) <= binsKeys[0]
	})

	//last bin
	bins[binsKeys[len(binsKeys)-1]] = filterRows(data, func(e linearalgebra.Matrix) bool {
		return e.Get(0, 0) > binsKeys[len(binsKeys)-1] //ADD A  NEW BIN AFTER THE LAST ELEMENTS
	})

	// middle bins
	var i int
	j := 1
	for j < len(binsKeys) {
		bins[binsKeys[j]] = filterRows(data, func(e linearalgebra.Matrix) bool {
			return (e.Get(0, 0) > binsKeys[i]) && (e.Get(0, 0) <= binsKeys[j])
		})
		i++
		j++
	}

	return bins
}

func giniImpurity(data linearalgebra.Matrix) float64 {
	classValueCounts := getValueCounts(data.GetCol(1))
	var gini float64

	for _, value := range classValueCounts {
		pValue := float64(value) / float64(data.Row) // probability of getting this class
		gini += pValue * pValue
	}

	return 1 - gini
}

func featuresGini(data linearalgebra.Matrix) float64 {

	midPoints := getMidPoints(getUniqueValues(data.GetCol(0))) // mid points bin for this feature
	featureBins := extractBins(data, midPoints)                // map of features bins where the keys are the midpoints

	totalElements := data.Row // total number of elements in the feature
	var tGini float64         //total gini impurity of the feature
	for _, bin := range featureBins {
		// binClassValueCounts := getValueCounts(bin.GetCol(0))
		tGini += (float64(bin.Row) / float64(totalElements)) * giniImpurity(bin)
	}

	return tGini
}

func selectSplit(features, target linearalgebra.Matrix) int {

	currentFeatureImp := 80.0       // current feature impurity initialized at 80.0 since its max value is 0.5
	var minImpurityFeatureIndex int //holds the column index of the feature with the minimun gini impurity
	for i := 0; i < features.Col; i++ {
		featureTarget := features.GetCol(i).InsertAt(target, 1)
		currentFeatureGini := featuresGini(featureTarget)
		if currentFeatureImp > currentFeatureGini {
			currentFeatureImp = currentFeatureGini
			minImpurityFeatureIndex = i
		}
	}
	return minImpurityFeatureIndex
}

// gets the unique values of a column vector
// target: is the vector from which the unique values will be extracted. It must be a column vector
// returns a column vector
func getUniqueValues(target linearalgebra.Matrix) linearalgebra.Matrix {
	if !isVector(target) {
		panic("target must be a column Vector")
	}

	values := make(map[float64]bool)
	uniqueClasses := make([]float64, 0, target.Row)

	for i := 0; i < target.Row; i++ {
		currentClass := target.Get(i, 0)
		_, present := values[currentClass]
		if !present {
			values[currentClass] = true
			uniqueClasses = append(uniqueClasses, currentClass)
		}
	}
	return linearalgebra.NewColumnVector(uniqueClasses)
}

// returns a slice of the midpoints between the sorted element of a vector
// data: is the vector to be proccessed. It must be a column vector
func getMidPoints(data linearalgebra.Matrix) []float64 {
	sort.Float64s(data.Data)
	r := append(data.Data, data.Data[len(data.Data)-1]+1) //adds extra element to obtain a midpoint above the last number
	fmt.Println(r)
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
	if v.Col == 1 {
		return true
	}
	return false
}

func filter(slice []float64, f func(e float64) bool) []float64 {
	newSlice := make([]float64, 0, len(slice))

	for _, e := range slice {
		if f(e) {
			newSlice = append(newSlice, e)
		}
	}
	return newSlice
}

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
