package CloudForest

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
)

func ParseLibSVM(input io.Reader) *FeatureMatrix {
	reader := bufio.NewReader(input)

	data := make([]Feature, 0, 100)
	lookup := make(map[string]int, 0)
	labels := make([]string, 0, 0)

	i := 0
	//ncases := 8100000
	//allthedata := make([]uint8, 784*ncases, 784*ncases)
	for {

		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		} else if err != nil {
			log.Print("Error:", err)
			return nil
		}

		vals := strings.Fields(line)

		if i == 0 {
			if strings.Contains(vals[0], ".") {
				//looks like a float...add dense float64 feature regression
				data = append(data, &DenseNumFeature{
					make([]float64, 0, 0),
					make([]bool, 0, 0),
					"0",
					false})

			} else {
				//doesn't look like a float...add dense catagorical
				data = append(data, &DenseCatFeature{
					&CatMap{make(map[string]int, 0),
						make([]string, 0, 0)},
					make([]int, 0, 0),
					make([]bool, 0, 0),
					"0",
					false,
					false})
			}
		}
		data[0].Append(vals[0])

		for _, v := range vals[1:] {
			parts := strings.Split(v, ":")
			xi, err := strconv.Atoi(parts[0])
			if err != nil {
				log.Print("Atoi error: ", err, " Line ", i, " Parsing: ", v)
			}
			//pad out the data to include this feature
			for xi >= len(data) {
				/*data = append(data, &SparseNumFeature{
					make(map[int]float64),
					make(map[int]bool),
					fmt.Sprintf("%v", len(data)),
					false})

				/*
					data = append(data, &DenseUint8Feature{
						allthedata[(len(data)-1)*ncases : len(data)*ncases],
						make(map[int]bool),
						fmt.Sprintf("%v", len(data))})
					/*
						data = append(data, &SparseUint8Feature{
							make(map[int]uint8),
							make(map[int]bool),
							fmt.Sprintf("%v", len(data))})
				*/

			}
			data[xi].PutStr(i, parts[1])

		}
		label := fmt.Sprintf("%v", i)
		lookup[label] = i
		labels = append(labels, label)

		i++

	}

	fm := &FeatureMatrix{data, lookup, labels}

	return fm

}
