package testutils

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/marti700/veritas/linearalgebra"
)

// Returns true if the difference between the expected and actual result are whith the delta parameter range
func AcceptableResults(expected, actual linearalgebra.Matrix, delta float64) bool {
	for i := range expected.Data {
		absoluteDelta := math.Abs(expected.Data[i] - actual.Data[i])
		if (delta <= absoluteDelta) || math.IsNaN(absoluteDelta) {
			return false
		}
	}
	return true
}

// Reads data from a csv file and returns the read data as a Matrix
// this functiona assumes the data in the csv are numbers in the float64 range
func ReadDataFromcsv(pathToFile string) linearalgebra.Matrix {
	f, err := os.Open(pathToFile)
	if err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(f)
	var matrixData string

	// read file first line to get the matrix column number
	// this are the heading numbers of the csv files
	// this line can be discarted since it does not hold useful data
	scanner.Scan()
	fstLine := scanner.Text()
	cols_num := len(strings.Split(fstLine, ","))

	// loop through the rest of the file
	fileLines := 0
	for scanner.Scan() {
		// extra coma so that the last number of this line don't get mixed with the first number of the next when slitting later
		matrixData += scanner.Text()+","
		fileLines++
	}

	dataSplit := strings.Split(matrixData, ",")

	matrix := make([][]float64, fileLines)
	row := 0
	col := 0
	nextBreak := cols_num

	data := make([]float64, cols_num)
	for i, e := range dataSplit {
		//before processing the matrix next row
		if i == nextBreak {
			matrix[row] = data
			data = make([]float64, cols_num)
			nextBreak += cols_num
			row++
			col = 0
		}
		data[col], _ = strconv.ParseFloat(e, 64)
		col++
	}
	return linearalgebra.NewMatrix(matrix)
}
