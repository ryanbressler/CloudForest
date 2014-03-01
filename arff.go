package CloudForest

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"strings"
)

func ParseARFF(input io.Reader) *FeatureMatrix {

	reader := bufio.NewReader(input)

	data := make([]Feature, 0, 100)
	lookup := make(map[string]int, 0)

	i := 0
	for {

		line, err := reader.ReadString('\n')
		if err != nil {
			log.Print("Error:", err)
			return nil
		}
		norm := strings.ToLower(line)

		if strings.HasPrefix(norm, "@data") {
			break
		}

		if strings.HasPrefix(norm, "@attribute") {
			vals := strings.Fields(line)

			if strings.ToLower(vals[2]) == "numeric" {
				data = append(data, &DenseNumFeature{
					make([]float64, 0, 0),
					make([]bool, 0, 0),
					vals[1]})
			} else {
				data = append(data, &DenseCatFeature{
					&CatMap{make(map[string]int, 0),
						make([]string, 0, 0)},
					make([]int, 0, 0),
					make([]bool, 0, 0),
					vals[1],
					false})
			}

			lookup[vals[1]] = i
			i++
		}

	}

	fm := &FeatureMatrix{data, lookup, make([]string, 0, 0)}

	csvdata := csv.NewReader(reader)
	//csvdata.Comma = ','

	fm.LoadCases(csvdata, false)
	return fm

}
