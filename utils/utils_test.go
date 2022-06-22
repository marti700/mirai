package utils

import (
	"testing"

	"github.com/marti700/mirai/testutils"
	"github.com/marti700/veritas/linearalgebra"
)


func TestBootstrap(t *testing.T) {

	data := testutils.ReadDataFromcsv("../testdata/datagenerators/data/cdecisiontree/data/x_train.csv")

	inBag, outOfBag := Bootstrap(data)

	if linearalgebra.IsEmpty(inBag) || linearalgebra.IsEmpty(outOfBag) {
		t.Error("one of inBag or outOfBag sets are empty")
	}
}