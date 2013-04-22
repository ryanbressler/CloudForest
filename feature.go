package CloudForest

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

/*CatMap is for mapping catagorical values to integers.
It contains:

	Map  : a map of ints by the string used fot the catagory
	Back : a slice of strings by the int that represents them
*/
type CatMap struct {
	Map  map[string]int //map categories from string to Num
	Back []string       // map categories from Num to string
}

//CatToNum provides the Num equivelent of the provided catagorical value
//if it allready exists or adds it to the map and returns the new value if 
//it doesn't.
func (cm *CatMap) CatToNum(value string) (numericv int) {
	numericv, exsists := cm.Map[value]
	if exsists == false {
		numericv = len(cm.Back)
		cm.Map[value] = numericv
		cm.Back = append(cm.Back, value)

	}
	return
}

/*Structure representing a single feature in a feature matrix.
It contains:
An embedded CatMap (may only be instantiated for cat data)
	NumData   : A slice of floates used for numerical data and nil otherwise
	CatData   : A slice of ints for catagorical data and nil otherwise
	Missing   : A slice of bools indicating missing values. Measure this for length.
	Numerical : is the feature numerical
	Name      : the name of the feature*/
type Feature struct {
	*CatMap
	NumData   []float64
	CatData   []int
	Missing   []bool
	Numerical bool
	Name      string
}

//ParseFeature parses a Feature from an array of strings and a capacity 
//capacity is the number of cases and will usually be len(record)-1 but
//but doesn't need to be calculated for every row of a large file.
//The type of the feature us infered from the start ofthe first (header) field 
//in record:
//"N:"" indicating numerical, anything else (usually "C:" and "B:") for catagorical
func ParseFeature(record []string, capacity int) Feature {
	f := Feature{
		&CatMap{make(map[string]int, 0),
			make([]string, 0, 0)},
		nil,
		nil,
		make([]bool, 0, capacity),
		false,
		record[0]}

	switch record[0][0:2] {
	case "N:":
		f.NumData = make([]float64, 0, capacity)
		//Numerical
		f.Numerical = true
		for i := 1; i < len(record); i++ {
			v, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				f.NumData = append(f.NumData, 0.0)
				f.Missing = append(f.Missing, true)
				continue
			}
			f.NumData = append(f.NumData, float64(v))
			f.Missing = append(f.Missing, false)

		}

	default:
		f.CatData = make([]int, 0, capacity)
		//Assume Catagorical
		f.Numerical = false
		for i := 1; i < len(record); i++ {
			v := record[i]
			norm := strings.ToLower(v)
			if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {

				f.CatData = append(f.CatData, 0)
				f.Missing = append(f.Missing, true)
				continue
			}
			f.CatData = append(f.CatData, f.CatToNum(v))
			f.Missing = append(f.Missing, false)

		}

	}
	return f

}

//BestSplit finds the best split of the features that can be achieved using the specified target and cases
//it returns a Splitter and the impurity
func (f *Feature) BestSplit(target *Feature, cases []int) (s *Splitter, impurity float64) {

	switch f.Numerical {
	case true:
		s = &Splitter{f.Name, true, 0.0, nil, nil}
		sortableCases := SortableFeature{f, cases}
		sort.Sort(sortableCases)
		for index, i := range sortableCases.Cases {
			l := sortableCases.Cases[:i]
			r := sortableCases.Cases[i:]
			//BUG(ryan) is this the proper way to combine impurities??
			innerimp := (target.Impurity(l) + target.Impurity(r)) / 2.0
			if index == 0 || innerimp < impurity {
				impurity = innerimp
				s.Value = f.NumData[i]

			}

		}
	case false:
		//BUG(ryan) find the best way to split a catagorical feature
	}
	return

}

//BUG(ryan) BestSplitter not done yet...relies on stubs
//Find the best splitter
func (target *Feature) BestSplitter(fm *FeatureMatrix, cases []int, mTry int) (s *Splitter) {
	//BUG(ryan) generate canidate features randomly or accept list of canidates
	canidates := make([]int, 0)
	bestImpurity := 0.0
	for index, i := range canidates {
		splitter, impurity := fm.Data[i].BestSplit(target, cases)
		if index == 0 || impurity < bestImpurity {
			bestImpurity = impurity
			s = splitter
		}

	}
	return
}

//Impurity returns Gini impurity or RMS vs the mean for a set of cases
//depending on weather the feature is catagorical or numerical
func (target *Feature) Impurity(cases []int) (e float64) {
	switch target.Numerical {
	case true:
		//BUG(ryan) is this the right way to calculate impurity for numericals???
		m := target.Mean(cases)
		e = target.RMS(cases, m)
	case false:
		e = target.Gini(cases)
	}
	return

}

//Gini returns the gini impurity for the specified cases in the feature
//gini impurity is calculated as 1 - Sum(fi^2) where fi is the fraction
//of cases in the ith catagory.
func (target *Feature) Gini(cases []int) (e float64) {
	counter := make(map[int]int)
	total := 0
	for _, i := range cases {
		if !target.Missing[i] {
			v := target.CatData[i]
			if _, ok := counter[v]; !ok {
				counter[v] = 0

			}
			counter[v] = counter[v] + 1
			total += 1
		}
	}
	e = 1.0
	t := float64(total * total)
	for _, v := range counter {
		e -= float64(v*v) / t
	}
	return
}

//RMS returns the Root Mean Square error of the cases specifed vs the predicted
//value
func (target *Feature) RMS(cases []int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range cases {
		if !target.Missing[i] {
			d := predicted - target.NumData[i]
			e += d * d
			n += 1
		}

	}
	e = math.Sqrt(e / float64(n))
	return

}

//MEAN returns the mean of the feature for the cases specified 
func (target *Feature) Mean(cases []int) (m float64) {
	m = 0.0
	n := 0
	for _, i := range cases {
		if !target.Missing[i] {
			m += target.NumData[i]
			n += 1
		}

	}
	m = m / float64(n)
	return

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
				d := f.NumData[i]
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
				v := f.Back[f.CatData[i]]
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
