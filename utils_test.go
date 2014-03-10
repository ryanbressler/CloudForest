package CloudForest

import (
	"encoding/csv"
	"io"
	"testing"
)

func TestSpareseCounter(t *testing.T) {
	sc := new(SparseCounter)
	sc.Add(1, 1, 1)

	pipereader, pipewriter := io.Pipe()

	go func() {
		sc.WriteTsv(pipewriter)
		pipewriter.Close()
	}()

	tsv := csv.NewReader(pipereader)
	tsv.Comma = '\t'

	records, err := tsv.Read()
	if err != nil {
		t.Errorf("Error reading tsv output by SpareCOunter %v", err)
	}
	if l := len(records); l != 3 {
		t.Error("Sparse counter output tsv with %v records", l)
	}
	for i, r := range records {
		if r != "1" {
			t.Errorf("Spares counter out put wrong value %v or field %v", r, i)
		}
	}

}

func TestParseAsIntOrFractionOfTotal(t *testing.T) {

	if p := ParseAsIntOrFractionOfTotal("70", 100); p != 70 {
		t.Errorf("ParseAsIntOrFractionOfTotal parsed 70 as %v", p)
	}

	if p := ParseAsIntOrFractionOfTotal(".7", 100); p != 70 {
		t.Errorf("ParseAsIntOrFractionOfTotal parsed .7 as %v / 100", p)
	}
}
