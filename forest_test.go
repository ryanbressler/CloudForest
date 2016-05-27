package CloudForest

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

var (
	predFilePath  = "preds.csv"
	inBagFilePath = "n.csv"
)

func TestPartial(t *testing.T) {
	irisreader := strings.NewReader(irislibsvm)
	fm := ParseLibSVM(irisreader)

	for i, feature := range fm.Data {
		t.Logf("feature %d: %s", i, feature.GetName())
	}

	tgt := fm.Data[0]
	model := GrowRandomForest(fm, tgt, &ForestConfig{
		NSamples: fm.Data[0].Length(),
		MTry:     2,
		NTrees:   500,
		LeafSize: 1,
	})

	forest := model.Forest
	x, y := P(forest, fm, "2")
	t.Log(x)
	t.Log(y)
}

func TestJackKnife(t *testing.T) {
	// read data
	preds := readCsv(t, predFilePath)
	inbag := readCsv(t, inBagFilePath)

	// run jackknife
	predictions, err := JackKnife(preds, inbag)
	if err != nil {
		t.Fatalf("error jack-knifing: %v", err)
	}

	if os.Getenv("EXPORT_JACKKNIFE") != "" {
		file, err := os.Create("validation.csv")
		if err != nil {
			t.Fatalf("error creating file: %v", err)
		}
		defer file.Close()

		fmt.Fprintln(file, "prediction, variance")
		for _, pred := range predictions {
			fmt.Fprintf(file, "%v, %v\n", pred.Value, pred.Variance)
		}
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
