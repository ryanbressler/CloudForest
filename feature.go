package CloudForest

import (
	"strconv"
	"strings"
)

type Num float64

type CatMap struct {
	Map     map[string]Num //map categories from string to Num
	Back    map[Num]string // map categories from Num to string
	nvalues Num
}

//CatToNum provides the Num equivelent of the provided catagorical value
//if it allready exists or adds it to the map and returns the new value if 
//it doesn't.
func (cm *CatMap) CatToNum(value string) (numericv Num) {
	numericv, exsists := cm.Map[value]
	if exsists == false {
		cm.Map[value] = cm.nvalues
		cm.Back[cm.nvalues] = value
		numericv = cm.nvalues
		cm.nvalues += 1.0
	}
	return
}

//Structure representing a single feature in a feature matrix.
//this data strucutre is related to the one in rf-ace which uses 
//the NUm/float64 data type for all feature values.
// In the future we may wish to represent
//catagorical features usings ints to speed up hash look up etc.
type Feature struct {
	*CatMap
	Data      []Num
	Missing   []bool
	Numerical bool
	Name      string
}

//Construct a NewFeature from an array of strings and a capacity 
//capacity is the number of cases and will usually be len(record)-1 but
//but doesn't need to be calculated for every row of a large file.
//The type of the feature us infered from the start ofthe first (header) field 
//in record:
//"N:"" indicating numerical, anything else (usually "C:" and "B:") for catagorical
func NewFeature(record []string, capacity int) Feature {
	f := Feature{
		&CatMap{make(map[string]Num, capacity),
			make(map[Num]string, capacity),
			0.0},
		make([]Num, 0, capacity),
		make([]bool, 0, capacity),
		false,
		record[0]}

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
		for i := 1; i < len(record); i++ {
			v := record[i]
			norm := strings.ToLower(v)
			if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {

				f.Data = append(f.Data, Num(0))
				f.Missing = append(f.Missing, true)
				continue
			}
			f.Data = append(f.Data, Num(f.CatToNum(v)))
			f.Missing = append(f.Missing, false)

		}

	}
	return f

}
