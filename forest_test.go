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

	if os.Getenv("WRITEDATA") != "" {
		iris, err := os.Create("iris.csv")
		if err != nil {
			t.Fatal(err)
		}

		for _, feature := range fm.Data {
			str := make([]string, feature.Length())
			for i := 0; i < feature.Length(); i++ {
				str[i] = feature.GetStr(i)
			}
			iris.WriteString(strings.Join(str, ","))
			iris.Write([]byte("\n"))
		}

		if err := iris.Close(); err != nil {
			t.Fatal(err)
		}
	}

	// make a good model
	tgt := fm.Data[0]
	model := GrowRandomForest(fm, tgt, &ForestConfig{
		NSamples: fm.Data[0].Length(),
		MTry:     3,
		NTrees:   500,
		LeafSize: 1,
	})

	forest := model.Forest
	x, y := PDP(forest, fm, "3")

	if os.Getenv("WRITEDATA") != "" {
		writeSlice("x.csv", x)
		writeSlice("y.csv", y)
	}
}

func writeSlice(name string, vals []float64) {
	f, _ := os.Create(name)
	str := make([]string, len(vals))
	for i, v := range vals {
		str[i] = strconv.FormatFloat(v, 'f', -1, 64)
	}

	f.WriteString(strings.Join(str, ","))
	f.Write([]byte("\n"))
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
