package CloudForest

import (
	"fmt"
	"math/rand"
	"sort"
)

type DenseNumFeature struct {
	NumData []float64
	Missing []bool
	Name    string
}

func (f *DenseNumFeature) NCats() int {
	return 0
}

func (f *DenseNumFeature) GetName() string {
	return f.Name
}

func (f *DenseNumFeature) Length() int {
	return len(f.Missing)
}

func (f *DenseNumFeature) IsMissing(i int) bool {
	return f.Missing[i]
}

func (f *DenseNumFeature) PutMissing(i int) {
	f.Missing[i] = true
}

func (f *DenseNumFeature) Get(i int) float64 {
	return f.NumData[i]
}

func (f *DenseNumFeature) Put(i int, v float64) {
	f.NumData[i] = v
	f.Missing[i] = false
}

func (f *DenseNumFeature) GoesLeft(i int, splitter *Splitter) bool {
	return f.NumData[i] <= splitter.Value
}

//Decode split builds a splitter from the numeric values returned by BestNumSplit or
//BestCatSplit. Numeric splitters are decoded to send values <= num left. Categorical
//splitters are decoded to send categorical values for which the bit in cat is 1 left.
func (f *DenseNumFeature) DecodeSplit(codedSplit interface{}) (s *Splitter) {

	s = &Splitter{f.Name, true, codedSplit.(float64), nil}

	return
}

/*
BestSplit finds the best split of the features that can be achieved using
the specified target and cases. It returns a Splitter and the decrease in impurity.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseNumFeature) BestSplit(target Target,
	cases *[]int,
	parentImp float64,
	allocs *BestSplitAllocs) (codedSplit interface{}, impurityDecrease float64) {

	*allocs.NonMissing = (*allocs.NonMissing)[0:0]
	*allocs.Right = (*allocs.Right)[0:0]

	for _, i := range *cases {
		if f.Missing[i] {
			*allocs.Right = append(*allocs.Right, i)
		} else {
			*allocs.NonMissing = append(*allocs.NonMissing, i)
		}
	}
	if len(*allocs.NonMissing) == 0 {
		return
	}
	nmissing := float64(len(*allocs.Right))
	total := float64(len(*cases))
	nonmissing := total - nmissing

	nonmissingparentImp := target.Impurity(allocs.NonMissing, allocs.Counter)

	missingimp := 0.0
	if nmissing > 0 {
		missingimp = target.Impurity(allocs.Right, allocs.Counter)
	}

	codedSplit, impurityDecrease = f.BestNumSplit(target, allocs.NonMissing, nonmissingparentImp, allocs)

	if nmissing > 0 {
		impurityDecrease = parentImp + ((nonmissing*(impurityDecrease-nonmissingparentImp) - nmissing*missingimp) / total)
	}
	return

}

/*
BestNumSplit searches over the possible splits of cases that can be made with f
and returns the one that minimizes the impurity of the target and the impurity decrease.

It expects to be provided for cases fir which the feature is not missing.

It searches by sorting the cases by the potential splitter and then evaluating each "gap"
between cases with non equal value as a potential split.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseNumFeature) BestNumSplit(target Target,
	cases *[]int,
	parentImp float64,
	allocs *BestSplitAllocs) (codedSplit interface{}, impurityDecrease float64) {

	impurityDecrease = minImp
	codedSplit = 0.0

	sorter := allocs.Sorter
	sorter.Feature = f
	sorter.Cases = *cases
	sort.Sort(sorter)

	// Note: timsort is slower for my test cases but could potentially be made faster by eliminating
	// repeated allocations

	for i := 1; i < len(sorter.Cases)-1; i++ {
		c := sorter.Cases[i]
		//skip cases where the next sorted case has the same value as these can't be split on
		if f.Missing[c] == true || f.NumData[c] == f.NumData[sorter.Cases[i+1]] {
			continue
		}

		/*		BUG there is a reallocation of a slice (not the underlying array) happening here in
				BestNumSplit accounting for a chunk of runtime. Tried copying data between *l and *r
				but it was slower.  */
		innerimp := parentImp - target.SplitImpurity(sorter.Cases[:i], sorter.Cases[i:], allocs.Counter)

		if innerimp > impurityDecrease {
			impurityDecrease = innerimp
			codedSplit = f.NumData[c]

		}

	}
	return
}

/*
FilterMissing loops over the cases and appends them into filtered.
For most use cases filtered should have zero length before you begin as it is not reset
internally
*/
func (f *DenseNumFeature) FilterMissing(cases *[]int, filtered *[]int) {
	for _, c := range *cases {
		if f.Missing[c] != true {
			*filtered = append(*filtered, c)
		}
	}

}

/*
SplitImpurity calculates the impurity of a split into the specified left and right
groups. This is defined as pLi*(tL)+pR*i(tR) where pL and pR are the probability of case going left or right
and i(tl) i(tR) are the left and right impurities.

Counter is only used for categorical targets and should have the same length as the number of categories in the target.
*/
func (target *DenseNumFeature) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	// l := *left
	// r := *right
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.NumImp(&l)
	impurityDecrease += nr * target.NumImp(&r)

	impurityDecrease /= nl + nr
	return
}

//Impurity returns Gini impurity or mean squared error vs the mean for a set of cases
//depending on weather the feature is categorical or numerical
func (target *DenseNumFeature) Impurity(cases *[]int, counter *[]int) (e float64) {

	e = target.NumImp(cases)

	return

}

//Numerical Impurity returns the mean squared error vs the mean calculated with a two pass algorythem.
func (target *DenseNumFeature) NumImp(cases *[]int) (e float64) {
	//TODO: benchmark regression with single pass computatioanl formula for sum of squares sum(xi^2)-sum(xi)^2/N
	//and/or with on-line algorithm http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
	m := target.Mean(cases)
	e = target.MeanSquaredError(cases, m)
	return
}

//MeanSquaredError returns the  Mean Squared error of the cases specified vs the predicted
//value. Only non missing cases are considered.
func (target *DenseNumFeature) MeanSquaredError(cases *[]int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range *cases {
		if !target.Missing[i] {
			d := predicted - target.NumData[i]
			e += d * d
			n += 1
		}

	}
	e = e / float64(n)
	return

}

//Mean returns the mean of the feature for the cases specified
func (target *DenseNumFeature) Mean(cases *[]int) (m float64) {
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

//Mode returns the mode category feature for the cases specified
func (f *DenseNumFeature) Mode(cases *[]int) (m float64) {
	counts := make(map[float64]int)
	for _, i := range *cases {
		if !f.Missing[i] {
			counts[f.NumData[i]] += 1
		}

	}
	max := 0
	for k, v := range counts {
		if v > max {
			m = k
			max = v
		}
	}
	return

}

//Find predicted takes the indexes of a set of cases and returns the
//predicted value. For categorical features this is a string containing the
//most common category and for numerical it is the mean of the values.
func (f *DenseNumFeature) FindPredicted(cases []int) (pred string) {

	pred = fmt.Sprintf("%v", f.Mean(&cases))

	return

}

/*ShuffledCopy returns a shuffled version of f for use as an artificial contrast in evaluation of
importance scores. The new feature will be named featurename:SHUFFLED*/
func (f *DenseNumFeature) ShuffledCopy() Feature {
	capacity := len(f.Missing)
	fake := &DenseNumFeature{
		nil,
		make([]bool, capacity),
		f.Name + ":SHUFFLED"}

	copy(fake.Missing, f.Missing)

	fake.NumData = make([]float64, capacity)
	copy(fake.NumData, f.NumData)

	//shuffle
	for j := 0; j < capacity; j++ {
		sourcei := j + rand.Intn(capacity-j)
		missing := fake.Missing[j]
		fake.Missing[j] = fake.Missing[sourcei]
		fake.Missing[sourcei] = missing

		data := fake.NumData[j]
		fake.NumData[j] = fake.NumData[sourcei]
		fake.NumData[sourcei] = data

	}
	return fake

}

//ImputeMissing imputes the missing values in a feature to the mean or mode of the feature.
func (f *DenseNumFeature) ImputeMissing() {
	cases := make([]int, 0, len(f.Missing))
	for i, _ := range f.Missing {
		cases = append(cases, i)
	}
	mean := 0.0

	mean = f.Mean(&cases)

	for i, m := range f.Missing {
		if m {

			f.NumData[i] = mean

			f.Missing[i] = false

		}
	}
}
