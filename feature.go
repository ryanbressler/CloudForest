package CloudForest

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

const maxExhaustiveCatagoires = 10

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

/*Feature is a structure representing a single feature in a feature matrix.
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

/*
BestCatSplit should be called on a catagorical feature to find the split that minimizes impurity
in the specified target.

rf-ace finds the "best" catagorical split using a greedy method that starts with the
single best catagory, and finds the best catagory to add on each iteration.

This implementation follows Brieman's implementation and the R/Matlab implementations 
based on it use exsaustive search overfor when there are less thatn 25/10 catagories 
and random splits above that.

Searching is implmented via bitwise oporations vs an incrementing or random integer for speed
but will currentlly only work when there are less catagories then the number of bits in an
int.
*/
func (f *Feature) BestCatSplit(target *Feature, cases *[]int, l *[]int, r *[]int, counter *[]int) (s *Splitter, impurityDecrease float64) {
	impurityDecrease = 0.0
	left := *l
	right := *r
	/*BUG(ryan) double check this is an exahustive search to find the best combination
	of catagories and make work for n > 64 (bigint? iterative?) search for nCats>10

	Finding the best catagorical split uses optimized code that proposes splits as
	integers abd uses bitwise oporations to find the best split
	*/
	nCats := len(f.Back)

	useExhaustive := nCats <= maxExhaustiveCatagoires
	nPartitions := 1
	if useExhaustive {
		//2**(nCats-2) is the number of valid partitions (collapsing symetric partions)
		nPartitions = (2 << uint(nCats-2))
	} else {
		//if more then the max just do the max randomly
		nPartitions = (2 << uint(maxExhaustiveCatagoires-2))
	}
	var bestSplit = 0
	//start at 1 to ingnore the set with all on one side
	for i := 1; i < nPartitions; i++ {

		bits := i
		if !useExhaustive {
			//generate random partition
			bits = rand.Int()
		}

		//check the value of the j'th bit of i and
		//send j left or right
		left = left[0:0]
		right = right[0:0]
		j := 0
		for _, c := range *cases {
			if f.Missing[c] == false {
				j = f.CatData[c]
				switch 0 != (bits & (1 << uint(j))) {
				case true:
					left = append(left, c)
				case false:
					right = append(right, c)
				}
			}

		}

		//skip cases where the split didn't do any splitting
		if len(left) == 0 || len(right) == 0 {
			continue
		}
		innerimp := target.ImpurityDecrease(&left, &right, counter)
		if innerimp > impurityDecrease {
			bestSplit = bits
			impurityDecrease = innerimp

		}

	}
	//Decode the best split and build a spliter from it
	if impurityDecrease > 0.0 {
		s = &Splitter{f.Name, false, 0.0, make(map[string]bool), make(map[string]bool)}

		for j := 0; j < nCats; j++ {

			switch 0 != (bestSplit & (1 << uint(j))) {
			case true:
				s.Left[f.Back[j]] = true
			case false:
				s.Right[f.Back[j]] = true
			}

		}

	}
	return
}

func (f *Feature) BestNumSplit(target *Feature, cases *[]int, l *[]int, r *[]int, counter *[]int) (s *Splitter, impurityDecrease float64) {
	impurityDecrease = 0.0
	left := *l
	right := *r
	bestSplit := 0.0
	sortableCases := SortableFeature{f, *cases}
	sort.Sort(sortableCases)
	for i := 1; i < len(sortableCases.Cases)-1; i++ {
		//skip cases where the next sorted case has the same value as these can't be split on
		if f.Missing[sortableCases.Cases[i]] == true || f.NumData[sortableCases.Cases[i]] == f.NumData[sortableCases.Cases[i+1]] {
			continue
		}
		innerSplit := &Splitter{f.Name, true, f.NumData[sortableCases.Cases[i]], nil, nil}
		left = left[0:0]
		right = right[0:0]
		innerSplit.SplitNum(f, cases, &left, &right)
		//inline the below to avoid realocating counters for gini impurity
		innerimp := target.ImpurityDecrease(&left, &right, counter)

		if innerimp > impurityDecrease {
			impurityDecrease = innerimp
			bestSplit = f.NumData[sortableCases.Cases[i]]

		}

	}
	if impurityDecrease > 0.0 {
		s = &Splitter{f.Name, true, bestSplit, nil, nil}
	}
	return
}

/*
BUG(ryan) BestSplit finds the best split of the features that can be achieved using 
the specified target and cases it returns a Splitter and the decrease in impurity
*/
func (f *Feature) BestSplit(target *Feature, cases *[]int, l *[]int, r *[]int, counter *[]int) (s *Splitter, impurityDecrease float64) {
	//BUG() Is removing the missing cases in BestSplit the right thing to do?
	switch f.Numerical {
	case true:
		s, impurityDecrease = f.BestNumSplit(target, cases, l, r, counter)
	case false:
		s, impurityDecrease = f.BestCatSplit(target, cases, l, r, counter)
	}
	return

}

/* 
Impurity Decrease calculates the decrease in impurity by spliting into the specified left and right
groups. This is depined as pLi*(tL)+pR*i(tR) where pL and pR are the probability of case going left or right
and i(tl) i(tR) are the left and right impurites.

Counter is only used for catagorical targets and should have length equal to the number of catagoires. It is
only included to avoid realocation and can be nil for non catagorical targets.
*/
func (target *Feature) ImpurityDecrease(left *[]int, right *[]int, counter *[]int) (impurityDecrease float64) {
	l := *left
	r := *right
	nl := float64(len(l))
	nr := float64(len(r))
	switch target.Numerical {
	case true:
		impurityDecrease = nl * target.NumImp(&l)
		impurityDecrease += nr * target.NumImp(&r)
	case false:
		impurityDecrease = nl * target.giniWithoutAlocate(&l, counter)
		impurityDecrease += nr * target.giniWithoutAlocate(&r, counter)
	}
	impurityDecrease /= nl + nr
	return
}

/*
BestSplitter finds the best splitter from a number of canidate features to 
slit on by looping over all features and calling BestSplit 
*/
func (target *Feature) BestSplitter(fm *FeatureMatrix, cases []int, canidates []int) (s *Splitter, impurityDecrease float64) {
	impurityDecrease = 0.0

	var counter []int

	if target.Numerical == false {
		counter = make([]int, len(target.Back), len(target.Back))
	}

	left := make([]int, len(cases))
	right := make([]int, len(cases))

	for _, i := range canidates {
		left = left[:]
		right = right[:]
		splitter, inerImp := fm.Data[i].BestSplit(target, &cases, &left, &right, &counter)
		if inerImp > impurityDecrease {
			impurityDecrease = inerImp
			s = splitter
		}

	}
	return
}

//Impurity returns Gini impurity or RMS vs the mean for a set of cases
//depending on weather the feature is catagorical or numerical
func (target *Feature) Impurity(cases *[]int) (e float64) {
	switch target.Numerical {
	case true:
		//BUG(ryan) is RMS vs the Mean the right definition of impurity for numerical groups?
		m := target.Mean(cases)
		e = target.RMS(cases, m)
	case false:
		e = target.Gini(cases)
	}
	return

}

func (target *Feature) NumImp(cases *[]int) (e float64) {
	m := target.Mean(cases)
	e = target.RMS(cases, m)
	return
}

//Gini returns the gini impurity for the specified cases in the feature
//gini impurity is calculated as 1 - Sum(fi^2) where fi is the fraction
//of cases in the ith catagory.
func (target *Feature) Gini(cases *[]int) (e float64) {
	counter := make([]int, len(target.Back))
	e = target.giniWithoutAlocate(cases, &counter)
	return
}

func (target *Feature) giniWithoutAlocate(cases *[]int, counts *[]int) (e float64) {
	total := 0
	counter := *counts
	for i, _ := range counter {
		counter[i] = 0
	}
	for _, i := range *cases {
		if !target.Missing[i] {
			counter[target.CatData[i]] += 1
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
func (target *Feature) RMS(cases *[]int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range *cases {
		if !target.Missing[i] {
			d := predicted - target.NumData[i]
			e += d * d
			n += 1
		}

	}
	e = math.Sqrt(e / float64(n))
	return

}

//Mean returns the mean of the feature for the cases specified 
func (target *Feature) Mean(cases *[]int) (m float64) {
	m = 0.0
	n := 0
	for _, i := range *cases {
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
		pred = fmt.Sprintf("%v", f.Mean(&cases))

	case false:
		//catagorical
		m := make([]int, len(f.Back))
		for _, i := range cases {
			if !f.Missing[i] {
				m[f.CatData[i]] += 1
			}

		}
		max := 0
		for k, v := range m {
			if v > max {
				pred = f.Back[k]
				max = v
			}
		}

	}
	return

}
