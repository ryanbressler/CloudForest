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
	predReader := csvReader(t, predFilePath)
	predStr, err := predReader.Read()
	if err != nil {
		t.Fatalf("could not read file %s: %v", predFilePath, err)
	}
	preds := strToFloat(t, predStr)
	t.Logf("length: %v", len(preds))

	inbagReader := csvReader(t, inBagFilePath)
	inbagStr, err := inbagReader.ReadAll()
	if err != nil {
		t.Fatalf("could not read file %s: %v", inBagFilePath, err)
	}
	inbag := make([][]float64, len(inbagStr))
	for i, v := range inbagStr {
		inbag[i] = strToFloat(t, v)
	}

	t.Logf("length: %v", inbag[0][4])

	// run jackknife
	mean, variance := JackKnife(preds, inbag)
	t.Logf("preds: %v, variance: %v", mean, variance)
}

func csvReader(t *testing.T, file string) *csv.Reader {
	predFile, err := os.Open(file)
	if err != nil {
		t.Fatalf("could not open file %s: %v", predFile, err)
	}

	return csv.NewReader(predFile)
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
