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

// Given data as a two column matrix, where the first column represents a feature and the second column the label for that feature
// and a set of float64 keys. Builds a map on which the data for any key is less than the key itself
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

// calculates the gini impurity of a feature
// feature must be given as a two column matrix where the first column represents the feature values
// and the last column the labels for the feature
func giniImpurity(data linearalgebra.Matrix) float64 {
	classValueCounts := getValueCounts(data.GetCol(1))
	var gini float64

	for _, value := range classValueCounts {
		pValue := float64(value) / float64(data.Row) // probability of getting this class
		gini += pValue * pValue
	}

	return 1 - gini
}

// Given a two column matrix where the first column represents the feature values and the last the labels of the feature
// returns a map on which the data for any key is less than the key itself
func getFeatureBins(data linearalgebra.Matrix) map[float64]linearalgebra.Matrix {
	midPoints := getMidPoints(getUniqueValues(data.GetCol(0))) // mid points bin for this feature
	return extractBins(data, midPoints)                        // map of features bins where the keys are the midpoints
}

// Calculates the total impurity for a given feature given as a two column matrix
// where the first column represents the feature values and the second the labels of the feature
func featuresGini(data linearalgebra.Matrix) float64 {
	featureBins := getFeatureBins(data) // map of features bins where the keys are the midpoints

	totalElements := data.Row // total number of elements in the feature
	var tGini float64         //total gini impurity of the feature
	for _, bin := range featureBins {
		tGini += (float64(bin.Row) / float64(totalElements)) * giniImpurity(bin)
	}

	return tGini
}

//finds the best sub-feature to be used for tree splitting
//This function selects the sub-feature with the lowest impurity, if two subfeatures have the same impurity
// the one with more rows is selected.
func bestFeatureBin(bins map[float64]linearalgebra.Matrix) float64 {
	//selected bin will hold the bin with the lowest impurity. It is initialized to 80.0 since impurity max value is 0.5
	sBin := 80.0
	impurities := make([]float64, 0, len(bins)+1)
	impurities = append(impurities, 42)
	keys := make([]float64, 0, len(bins)+1)
	keys = append(keys, 42)
	for key, bin := range bins {
		currentBinImpurity := giniImpurity(bin)
		if impurities[len(impurities)-1] > currentBinImpurity {
			sBin = key
		}
		if impurities[len(impurities)-1] == currentBinImpurity {
			if bins[keys[len(keys)-1]].Row < bin.Row {
				sBin = key
			}
		}
		impurities = append(impurities, currentBinImpurity)
		keys = append(keys, key)
	}
	return sBin
}

// Returns the index of the best splitting feature, given the features as a matrix and the labels as a vector
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

func Train(data, target linearalgebra.Matrix) {
	// 1- Find best feature split
	bestFeature := selectSplit(data, target)
	// 2- find best sub-feature split (based on midpoints)
	bFeatBin := bestFeatureBin(getFeatureBins(data.GetCol(bestFeature).InsertAt(target, 1)))
	fmt.Println(bFeatBin)
	// 3- split the data based in 1 and 2 (to get the left and right leaves of the tree)
	// 4- recursivly build the tree
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

// given a matrix and a boolean function of the type matrix -> bool returns a new matrix for which the function returns true
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

func filterRows2(data linearalgebra.Matrix, f func(r linearalgebra.Matrix) bool) (linearalgebra.Matrix, linearalgebra.Matrix) {
	var m1 linearalgebra.Matrix
	var m2 linearalgebra.Matrix
	for i := 0; i < data.Row; i++ {
		currentRow := data.GetRow(i)
		if f(currentRow) && len(m1.Data) == 0 {
			m1 = currentRow
		} else if !f(currentRow) && len(m2.Data) == 0 {
			m1 = currentRow
		} else if f(currentRow) {
			m1 = m1.InsertAt(currentRow, -1)
		} else {
			m2 = m2.InsertAt(currentRow, 0)
		}
	}
	return m1, m2
}