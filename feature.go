package CloudForest

import (
	"fmt"
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

//BUG(ryan) not done yet...just a stub
//Find the best splitter
func (target *Feature) BestSplitter(fm *FeatureMatrix, cases []int, mTry int) (s *Splitter) {
	switch target.Numerical {
	case true:
		for i, v := range fm.Data {

		}
	case false:
	}
	return
}

//Gini returns the gini impurity for the specified cases in the feature
//gini impurity is calculated as 1 - Sum(fi^2) where fi is the fraction
//of cases in the ith catagory.
func (target *Feature) Gini(cases []int) (e float64) {
	counter := make(map[Num]int)
	total := 0
	for _, i := range cases {
		if !target.Missing[i] {
			v := Num(target.Data[i])
			if _, ok := counter[v]; !ok {
				counter[v] = 0

			}
			counter[v] = counter[v] + 1
			total += 1
		}
	}
	e = 1.0
	total = float64(total * total)
	for _, v := range counter {
		e -= v * v / total
	}
}

//RMS returns the Root Mean Square error of the cases specifed vs the predicted
//value
func (target *Feature) RMS(cases []int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range cases {
		if !target.Missing[i] {
			d := predicted - float64(target.Data[i])
			e += d * d
			n += 1
		}

	}
	e = math.Sqrt(e / float64(n))

}

//MEAN returns the mean of the feature for the cases specified 
func (target *Feature) Mean(cases []int) (e float64) {
	e = 0.0
	n := 0
	for _, i := range cases {
		if !target.Missing[i] {
			d := predicted - float64(target.Data[i])
			e += d * d
			n += 1
		}

	}
	e = math.Sqrt(e / float64(n))

}

//Find predicted takes the indexes of a set of cases and returns the 
//predicted value. For catagorical features this is a string containing the
//most common catagory and for numerical it is the mean of the values.
func (f *Feature) FindPredicted(cases []int) (pred string) {
	switch f.Numerical {
	case true:
		//numerical
		v := 0.0
		count := 0
		for _, i := range cases {
			if !f.Missing[i] {
				d := f.Data[i]
				v += float64(d)
				count += 1
			}

		}
		pred = fmt.Sprintf("%v", v/float64(count))

	case false:
		//catagorical
		m := make(map[string]int)
		for _, i := range cases {
			if !f.Missing[i] {
				v := f.Back[f.Data[i]]
				if _, ok := m[v]; !ok {
					m[v] = 0
				}
				m[v] += 1
			}

		}
		max := 0
		for k, v := range m {
			if v > max {
				pred = k
				max = v
			}
		}

	}
	return

}
