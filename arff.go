package CloudForest

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"strings"
)

//ParseARFF reads a file in weka'sarff format:
//http://www.cs.waikato.ac.nz/ml/weka/arff.html
//The relation is ignored and only catagorical and numerical variables are supported
func ParseARFF(input io.Reader) *FeatureMatrix {

	reader := bufio.NewReader(input)

	data := make([]Feature, 0, 100)
	lookup := make(map[string]int, 0)
	//labels := make([]string, 0, 0)

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

			if strings.ToLower(vals[2]) == "numeric" || strings.ToLower(vals[2]) == "real" {
				data = append(data, &DenseNumFeature{
					make([]float64, 0, 0),
					make([]bool, 0, 0),
					vals[1],
					false})
			} else {
				data = append(data, &DenseCatFeature{
					&CatMap{make(map[string]int, 0),
						make([]string, 0, 0)},
					make([]int, 0, 0),
					make([]bool, 0, 0),
					vals[1],
					false,
					false})
			}

			lookup[vals[1]] = i
			//labels = append(labels, vals[1])
			i++
		}

	}

	fm := &FeatureMatrix{data, lookup, make([]string, 0, 0)}

	csvdata := csv.NewReader(reader)
	csvdata.Comment = '%'
	//csvdata.Comma = ','

	fm.LoadCases(csvdata, false)
	return fm

}

//WriteArffCases writes the specified cases from the provied feature matrix into an arff file with the given relation string.
func WriteArffCases(data *FeatureMatrix, cases []int, relation string, outfile io.Writer) error {
	/*@RELATION iris

	  @ATTRIBUTE sepallength  NUMERIC
	  @ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}*/

	fmt.Fprintf(outfile, "@RELATION %v\n\n", relation)

	for _, f := range data.Data {
		ftype := "NUMERIC"
		switch f.(type) {
		case (*DenseCatFeature):
			ftype = fmt.Sprintf("{%v}", strings.Join(f.(*DenseCatFeature).Back, ","))
		}

		fmt.Fprintf(outfile, "@ATTRIBUTE %v %v\n", f.GetName(), ftype)
	}

	fmt.Fprint(outfile, "\n@DATA\n")

	oucsv := csv.NewWriter(outfile)
	oucsv.Comma = ','

	for _, i := range cases {
		entries := make([]string, 0, 10)

		for _, f := range data.Data {
			v := "?"
			if !f.IsMissing(i) {
				v = f.GetStr(i)
			}
			entries = append(entries, v)

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
