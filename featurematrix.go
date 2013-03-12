package CloudForest

import (
	"encoding/csv"
	"io"
	"log"
	"strconv"
	"strings"
)

type Num float64

//this data strucutre is related to the one in rf-ace. In the future we may wish to represent
//catagorical features usings ints to speed up hash look up etc.
type Feature struct {
	Data      []Num
	Missing   []bool
	Numerical bool
	Map       map[string]Num //map categories from string to integer
	Back      map[Num]string // map categories from integer to string
	Name      string
}

func NewFeature(record []string, capacity int) Feature {
	f := Feature{make([]Num, 0, capacity), make([]bool, 0, capacity), false, make(map[string]Num, capacity), make(map[Num]string, capacity), record[0]}
	switch record[0][0:2] {
	case "N:":
		//Numerical
		f.Numerical = true
		for i := 1; i < len(record); i++ {
			v, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				f.Data = append(f.Data, Num(0))
				f.Missing = append(f.Missing, true)
				continue
			}
			f.Data = append(f.Data, Num(v))
			f.Missing = append(f.Missing, false)

		}

	default:
		//Assume Catagorical
		f.Numerical = false
		fvalue := Num(0.0)
		for i := 1; i < len(record); i++ {
			v := record[i]
			norm := strings.ToLower(v)
			if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {
				f.Data = append(f.Data, Num(0))
				f.Missing = append(f.Missing, true)
				continue
			}
			nv, exsists := f.Map[v]
			if exsists == false {
				f.Map[v] = fvalue
				f.Back[fvalue] = v
				nv = fvalue
				fvalue += 1.0
			}
			f.Data = append(f.Data, Num(nv))
			f.Missing = append(f.Missing, false)

		}

	}
	return f

}

type FeatureMatrix struct {
	Data []Feature
	Map  map[string]int
}

func ParseAFM(input io.Reader) *FeatureMatrix {
	data := make([]Feature, 0, 100)
	lookup := make(map[string]int, 0)
	tsv := csv.NewReader(input)
	tsv.Comma = '\t'
	_, err := tsv.Read()
	if err == io.EOF {
		return &FeatureMatrix{data, lookup}
	} else if err != nil {
		log.Print("Error:", err)
		return &FeatureMatrix{data, lookup}
	}
	capacity := tsv.FieldsPerRecord

	count := 0
	for {
		record, err := tsv.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Print("Error:", err)
			break
		}
		data = append(data, NewFeature(record, capacity))
		lookup[record[0]] = count
		count++
	}
	return &FeatureMatrix{data, lookup}
}
