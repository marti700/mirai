package testutils

func AcceptableReslts(expected, actual linearalgebra.Matrix) bool {
	for i := range expected.Data {
		if math.Abs(expected.Data[i]-actual.Data[i]) > 2 {
			return false
		}
	}
	return true
}