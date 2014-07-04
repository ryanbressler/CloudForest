package CloudForest

import (
	"bufio"
	"encoding/csv"
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
	ncases := 0
	for {
		ncases++

		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		} else if err != nil {
			log.Print("Error:", err)
			return nil
		}

		vals := strings.Fields(line)

		if i == 0 {
			name := "0"
			lookup[name] = 0
			if strings.Contains(vals[0], ".") {
				//looks like a float...add dense float64 feature regression
				data = append(data, &DenseNumFeature{
					make([]float64, 0, 0),
					make([]bool, 0, 0),
					name,
					false})

			} else {
				//doesn't look like a float...add dense catagorical
				data = append(data, &DenseCatFeature{
					&CatMap{make(map[string]int, 0),
						make([]string, 0, 0)},
					make([]int, 0, 0),
					make([]bool, 0, 0),
					name,
					false,
					false})
			}

		}
		data[0].Append(vals[0])

		//pad existing features
		for _, f := range data[1:] {
			f.Append("0")
		}

		for _, v := range vals[1:] {
			parts := strings.Split(v, ":")
			xi, err := strconv.Atoi(parts[0])
			if err != nil {
				log.Print("Atoi error: ", err, " Line ", i, " Parsing: ", v)
			}
			//pad out the data to include this feature
			for xi >= len(data) {
				name := fmt.Sprintf("%v", len(data))
				lookup[name] = len(data)
				data = append(data, &DenseNumFeature{
					make([]float64, ncases, ncases),
					make([]bool, ncases, ncases),
					name,
					false})

			}
			data[xi].PutStr(i, parts[1])

		}

		label := fmt.Sprintf("%v", i)
		labels = append(labels, label)
		i++

	}

	fm := &FeatureMatrix{data, lookup, labels}

	return fm

}

func WriteLibSvm(data *FeatureMatrix, targetn string, outfile io.Writer) error {
	targeti, ok := data.Map[targetn]
	if !ok {
		return fmt.Errorf("Target '%v' not found in data.", targetn)
	}
	target := data.Data[targeti]

	//data.Data = append(data.Data[:targeti], data.Data[targeti+1:]...)

	noTargetFm := &FeatureMatrix{make([]Feature, 0, len(data.Data)), make(map[string]int), data.CaseLabels}

	for i, f := range data.Data {
		if i != targeti {
			noTargetFm.Map[f.GetName()] = len(noTargetFm.Data)
			noTargetFm.Data = append(noTargetFm.Data, f.Copy())

		}
	}

	noTargetFm.ImputeMissing()
	encodedfm := noTargetFm.EncodeToNum()

	oucsv := csv.NewWriter(outfile)
	oucsv.Comma = ' '

	for i := 0; i < target.Length(); i++ {
		entries := make([]string, 0, 10)
		switch target.(type) {
		case NumFeature:
			entries = append(entries, target.GetStr(i))
		case CatFeature:
			entries = append(entries, fmt.Sprintf("%v", target.(CatFeature).Geti(i)))
		}

		for j, f := range encodedfm.Data {
			v := f.(NumFeature).Get(i)
			if v != 0.0 {
				entries = append(entries, fmt.Sprintf("%v:%v", j+1, v))
			}
		}
		//fmt.Println(entries)
		err := oucsv.Write(entries)
		if err != nil {
			return err
		}

	}
	oucsv.Flush()
	return nil
}

func WriteLibSvmCases(data *FeatureMatrix, cases []int, targetn string, outfile io.Writer) error {
	targeti, ok := data.Map[targetn]
	if !ok {
		return fmt.Errorf("Target '%v' not found in data.", targetn)
	}
	target := data.Data[targeti]

	noTargetFm := &FeatureMatrix{make([]Feature, 0, len(data.Data)), make(map[string]int), data.CaseLabels}

	encode := false
	for i, f := range data.Data {
		if i != targeti {
			if data.Data[i].NCats() > 0 {
				encode = true
			}
			noTargetFm.Map[f.GetName()] = len(noTargetFm.Data)
			noTargetFm.Data = append(noTargetFm.Data, f)

		}
	}

	noTargetFm.ImputeMissing()

	encodedfm := noTargetFm
	if encode {
		encodedfm = noTargetFm.EncodeToNum()
	}

	oucsv := csv.NewWriter(outfile)
	oucsv.Comma = ' '

	for _, i := range cases {
		entries := make([]string, 0, 10)
		switch target.(type) {
		case NumFeature:
			entries = append(entries, fmt.Sprintf("%g", target.(NumFeature).Get(i)))
		case CatFeature:
			entries = append(entries, fmt.Sprintf("%v", target.(CatFeature).Geti(i)))
		}

		for j, f := range encodedfm.Data {
			v := f.(NumFeature).Get(i)
			if v != 0.0 {
				entries = append(entries, fmt.Sprintf("%v:%v", j+1, v))
			}
		}
		//fmt.Println(entries)
		err := oucsv.Write(entries)
		if err != nil {
			return err
		}

	}
	oucsv.Flush()
	return nil
}
