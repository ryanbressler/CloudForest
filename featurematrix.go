package CloudForest

import (
	"encoding/csv"
	"io"
	"log"
)

//FeatureMatrix contains a slice of Features and a Map to look of the index of a feature
//by its string id.
type FeatureMatrix struct {
	Data []Feature
	Map  map[string]int
}

//Parse an AFM (anotated feature matrix) out of an io.Reader
//AFM format is a tsv with row and column headers where the row headers start with
//N: indicating numerical, C: indicating catagorical or B: indicating boolean
//For this parser features without N: are assumed to be catagorical
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
