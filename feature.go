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
const minImp = 1e-12

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
func ParseFeature(record []string) Feature {
	capacity := len(record)
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

/*IterBestCatSplit performs an iterative search to find the split that minimizes impurity
in the specified target.

rf-ace finds the "best" catagorical split using a greedy method that starts with the
single best catagory, and finds the best catagory to add on each iteration.


Searching is implmented via bitwise on intergers for speed but will currentlly only work
when there are less catagories then the number of bits in an int.

Pointers to slices for l and r and counter are used to reduce realocations during search
and will not contain meaningfull results.

l and r should have the same capacity as cases . counter is only used for catagorical targets and
should have the same length as the number of catagories in the target.*/
func (f *Feature) IterBestCatSplit(target *Feature, cases *[]int, l *[]int, r *[]int, counter *[]int) (bestSplit int, impurityDecrease float64) {

	left := *l
	right := *r
	/*BUG(ryan) this won't work for n > 32. Should use bigint? or iterative search for nCats>10

	This is an iterative search for the best combinations of catagories.
	*/

	nCats := len(f.Back)
	cat := 0

	//overall running best impurity and split
	impurityDecrease = minImp
	bestSplit = 0

	//running best with n catagories
	innerImp := minImp
	innerSplit := 0

	//values for the current proposed n+1 catagory
	nextImp := minImp
	nextSplit := 0

	//iterativelly build a combination of catagories untill they
	//stop getting better
	for j := 0; j < nCats; j++ {

		innerImp = minImp
		innerSplit = 0
		//find the best additonal catagory
		for i := 0; i < nCats; i++ {

			if 0 != (bestSplit & (1 << uint(i))) {
				continue
			}

			left = left[0:0]
			right = right[0:0]

			nextSplit = bestSplit | 1<<uint(i)

			for _, c := range *cases {
				if f.Missing[c] == false {
					cat = f.CatData[c]
					if 0 != (nextSplit & (1 << uint(cat))) {
						left = append(left, c)
					} else {
						right = append(right, c)
					}
				}

			}

			//skip cases where the split didn't do any splitting
			if len(left) == 0 || len(right) == 0 {
				continue
			}

			nextImp = target.ImpurityDecrease(&left, &right, counter)

			if nextImp > innerImp {
				innerSplit = nextSplit
				innerImp = nextImp

			}

		}
		if innerImp > impurityDecrease {
			bestSplit = innerSplit
			impurityDecrease = innerImp

		} else {
			break
		}

	}

	return
}

/*
BestCatSplit performs an exahustive or stoicastic search for the split that minimizes impurity
in the specified target.

This implementation follows Brieman's implementation and the R/Matlab implementations
based on it use exsaustive search overfor when there are less thatn 25/10 catagories
and random splits above that.

Searching is implmented via bitwise oporations vs an incrementing or random integer for speed
but will currentlly only work when there are less catagories then the number of bits in an
int.

Pointers to slices for l and r and counter are used to reduce realocations during search
and will not contain meaningfull results.

l and r should have the same capacity as cases . counter is only used for catagorical targets and
should have the same length as the number of catagories in the target.
*/
func (f *Feature) BestCatSplit(target *Feature,
	cases *[]int,
	l *[]int,
	r *[]int,
	counter *[]int) (bestSplit int, impurityDecrease float64) {

	impurityDecrease = minImp
	left := *l
	right := *r
	/*BUG(ryan) this won't work for n > 32. Should use bigint random or iterative search for nCats>10

	Eahustive search of combinations of catagories is carried out by iterating an Int and using
	the bits to define which catagories go to the left of the split.

	*/
	nCats := len(f.Back)

	useExhaustive := nCats <= maxExhaustiveCatagoires
	nPartitions := 1
	if useExhaustive {
		//2**(nCats-2) is the number of valid partitions (collapsing symetric partions)
		nPartitions = (2 << uint(nCats-2))
	} else {
		//if more then the max we will loop max times and generate random combinations
		nPartitions = (2 << uint(maxExhaustiveCatagoires-2))
	}
	bestSplit = 0
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

	return
}

//Decode split builds a sliter from the numeric values returned by BestNumSplit or
//BestCatSplit.
func (f *Feature) DecodeSplit(num float64, cat int) (s *Splitter) {
	if f.Numerical {
		s = &Splitter{f.Name, true, num, nil}
	} else {
		nCats := len(f.Back)
		s = &Splitter{f.Name, false, 0.0, make(map[string]bool, nCats)}

		for j := 0; j < nCats; j++ {

			if 0 != (cat & (1 << uint(j))) {
				s.Left[f.Back[j]] = true
			}

		}
	}
	return
}

/*
BestNumSsplit searches over the possible splits of cases that can be made with f
and returns the one that minimizes the impurity of the target and the impurity decrease.

It searches by sorting the cases by the potential splitter and then evaluating each "gap"
between cases with non equal value as a potential split.

Pointers to slices for l and r and counter are used to reduce realocations during search
and will not contain meaningfull results.

l and r should have the same capacity as cases . counter is only used for catagorical targets and
should have the same length as the number of catagories in the target.
*/
func (f *Feature) BestNumSplit(target *Feature, cases *[]int, l *[]int, r *[]int, counter *[]int, sorter *SortableFeature) (bestSplit float64, impurityDecrease float64) {
	impurityDecrease = minImp
	left := *l
	right := *r
	bestSplit = 0.0

	//sortableCases := SortableFeature{f, *cases}
	sorter.Feature = f
	sorter.Cases = *cases
	sort.Sort(sorter)
	for i := 1; i < len(sorter.Cases)-1; i++ {
		c := sorter.Cases[i]
		//skip cases where the next sorted case has the same value as these can't be split on
		if f.Missing[c] == true || f.NumData[c] == f.NumData[sorter.Cases[i+1]] {
			continue
		}

		/*
			The two loops below filter missing values from the left and right side.
			A slower but simpler version of the two for loops bellow would be:
			innerSplit := &Splitter{f.Name, true, f.NumData[c], nil, nil}
			left = left[0:0]
			right = right[0:0]
			innerSplit.SplitNum(f, cases, &left, &right)
		*/
		left = left[0:0]
		right = right[0:0]
		for _, j := range sorter.Cases[:i] {
			if f.Missing[j] == false {
				left = append(left, j)
			}
		}
		for _, j := range sorter.Cases[i:] {
			if f.Missing[j] == false {
				right = append(right, j)
			}
		}

		innerimp := target.ImpurityDecrease(&left, &right, counter)

		if innerimp > impurityDecrease {
			impurityDecrease = innerimp
			bestSplit = f.NumData[c]

		}

	}
	return
}

/*
BUG(ryan) BestSplit finds the best split of the features that can be achieved using
the specified target and cases it returns a Splitter and the decrease in impurity.

Pointers to slices for l and r and counter are used to reduce realocations during search
and will not contain meaningfull results.

l and r should have the same capacity as cases . counter is only used for catagorical targets and
should have the same length as the number of catagories in the target.
*/
func (f *Feature) BestSplit(target *Feature,
	cases *[]int,
	itter bool,
	l *[]int,
	r *[]int,
	counter *[]int,
	sorter *SortableFeature) (bestNum float64, bestCat int, impurityDecrease float64) {

	//BUG() Is removing the missing cases in BestSplit the right thing to do?
	switch f.Numerical {
	case true:
		bestNum, impurityDecrease = f.BestNumSplit(target, cases, l, r, counter, sorter)
	case false:
		if itter || len(f.Back) > 4 {
			bestCat, impurityDecrease = f.IterBestCatSplit(target, cases, l, r, counter)
		} else {
			bestCat, impurityDecrease = f.BestCatSplit(target, cases, l, r, counter)
		}

	}
	return

}

/*
Impurity Decrease calculates the decrease in impurity by spliting into the specified left and right
groups. This is depined as pLi*(tL)+pR*i(tR) where pL and pR are the probability of case going left or right
and i(tl) i(tR) are the left and right impurites.

Pointers to slices for l and r and counter are used to reduce realocations during search
and will not contain meaningfull results.

l and r should have the same capacity as cases . counter is only used for catagorical targets and
should have the same length as the number of catagories in the target.
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
func (target *Feature) BestSplitter(fm *FeatureMatrix,
	cases []int,
	canidates []int,
	itter bool,
	l *[]int,
	r *[]int) (s *Splitter, impurityDecrease float64) {
	impurityDecrease = minImp

	var f, bestF *Feature
	var num, bestNum, inerImp float64
	var cat, bestCat int

	var counter []int
	sorter := SortableFeature{nil, nil}
	if target.Numerical == false {
		counter = make([]int, len(target.Back), len(target.Back))
	}

	left := *l
	right := *r

	for _, i := range canidates {
		left = left[:]
		right = right[:]
		f = &fm.Data[i]
		num, cat, inerImp = f.BestSplit(target, &cases, itter, &left, &right, &counter, &sorter)
		//BUG more stringent cutoff in BestSplitter?
		if inerImp > 0.0 && inerImp > impurityDecrease {
			bestF = f
			impurityDecrease = inerImp
			bestNum = num
			bestCat = cat
		}

	}
	if impurityDecrease > minImp {
		s = bestF.DecodeSplit(bestNum, bestCat)
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

/*
giniWithoutAlocate calculates gini impurity using the spupplied counter which must
be a slcie with length equal to the number of cases. This allows you to reduce allocations
but the counter will also contain per catagory counts.
*/
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
