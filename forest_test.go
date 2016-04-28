package CloudForest

import (
	"encoding/csv"
	"os"
	"strconv"
	"testing"
)

var (
	predFilePath  = "preds.csv"
	inBagFilePath = "n.csv"
)

func TestJackKnife(t *testing.T) {
	// read data
	preds := readCsv(t, predFilePath)
	inbag := readCsv(t, inBagFilePath)

	// run jackknife
	predictions, err := JackKnife(preds, inbag)
	if err != nil {
		t.Fatalf("error jack-knifing: %v", err)
	}

	if len(predictions) != 2 {
		t.Fatal("expected 2 mean/variance to be returned")
	}

	for i, pred := range predictions {
		t.Logf("(%dth) prediction: %+v", i, pred)
	}
}

func readCsv(t *testing.T, file string) [][]float64 {
	predFile, err := os.Open(file)
	if err != nil {
		t.Fatalf("could not open file %s: %v", predFile, err)
	}

	reader := csv.NewReader(predFile)
	all, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("could not read file %s: %v", file, err)
	}

	values := make([][]float64, len(all))
	for i, v := range all {
		values[i] = strToFloat(t, v)
	}
	return values
}

func strToFloat(t *testing.T, values []string) []float64 {
	f := make([]float64, len(values))
	var err error
	for i := range f {
		f[i], err = strconv.ParseFloat(values[i], 64)
		if err != nil {
			t.Fatalf("could not convert %s, %v", values[i], err)
		}
	}
	return f
}
