package CloudForest

import (
	"fmt"
	"log"
	"math/rand"
	//"sort"
	"math"
	"strconv"
)

//DenseNumFeature contains dense float64 training data, possibly with missing values.
type DenseNumFeature struct {
	NumData    []float64
	Missing    []bool
	Name       string
	HasMissing bool
}

func NewDenseNumFeature(name string) *DenseNumFeature {
	return &DenseNumFeature{
		Name:    name,
		NumData: make([]float64, 0),
		Missing: make([]bool, 0),
	}
}

//Append will parse and append a single value to the end of the feature. It is generally only used
//during data parseing.
func (f *DenseNumFeature) Append(v string) {
	fv, err := strconv.ParseFloat(v, 64)
	if err != nil {
		f.NumData = append(f.NumData, 0.0)
		f.Missing = append(f.Missing, true)
		f.HasMissing = true
		return
	}
	f.NumData = append(f.NumData, float64(fv))
	f.Missing = append(f.Missing, false)
}

//Less checks if the value of case i is less then the value of j.
func (f *DenseNumFeature) Less(i int, j int) bool {
	return f.NumData[i] < f.NumData[j]
}

//PutStr parses a string and puts it in the i'th position
func (f *DenseNumFeature) PutStr(i int, v string) {
	fv, err := strconv.ParseFloat(v, 64)
	if err != nil {
		f.Missing[i] = true
		f.HasMissing = true
		return
	}
	f.NumData[i] = float64(fv)
	f.Missing[i] = false
}

//NCats returns the number of catagories, 0 for numerical values.
func (f *DenseNumFeature) NCats() int {
	return 0
}

//GetName returns the name of the feature.
func (f *DenseNumFeature) GetName() string {
	return f.Name
}

//Length returns the length of the feature.
func (f *DenseNumFeature) Length() int {
	return len(f.Missing)
}

//IsMissing checks if the value for the i'th case is missing.
func (f *DenseNumFeature) IsMissing(i int) bool {
	return f.Missing[i]
}

//MissingVals checks if the feature has any missing values.
func (f *DenseNumFeature) MissingVals() bool {
	return f.HasMissing
}

//PutMissing set's the i'th value to be missing.
func (f *DenseNumFeature) PutMissing(i int) {
	f.Missing[i] = true
	f.HasMissing = true
}

//Get returns the value in the i'th posiiton. It doesn't check for missing values.
func (f *DenseNumFeature) Get(i int) float64 {
	return f.NumData[i]
}

//Get str returns the string representing the value in the i'th position. It returns NA if tehe value is missing.
func (f *DenseNumFeature) GetStr(i int) (value string) {
	if f.Missing[i] {
		return "NA"
	}
	return fmt.Sprintf("%v", f.NumData[i])
}

//Put inserts the value v into the i'th position of the feature.
func (f *DenseNumFeature) Put(i int, v float64) {
	f.NumData[i] = v
	f.Missing[i] = false
}

//GoesLeft checks if the i'th case goes left according to the supplied spliter.
func (f *DenseNumFeature) GoesLeft(i int, splitter *Splitter) bool {
	return f.NumData[i] <= splitter.Value
}

//Predicted returns the prediction (the mean) that should be made for the supplied cases.
func (f *DenseNumFeature) Predicted(cases *[]int) float64 {
	return f.Mean(cases)
}

//Norm defines the norm to use to tell how far the i'th case if from the value v
func (f *DenseNumFeature) Norm(i int, v float64) float64 {
	return math.Abs(f.NumData[i] - v)

}

//Split does an inplace slit from a coded split (a float64) and returns slices pointing into the origional cases slice.
func (f *DenseNumFeature) Split(codedSplit interface{}, cases []int) (l []int, r []int, m []int) {
	length := len(cases)

	lastleft := -1
	lastright := length
	swaper := 0

	//Move left cases to the start and right cases to the end so that missing cases end up
	//in between.
	var split float64
	switch s := codedSplit.(type) {
	case float64:
		split = s
	case int:
		split = float64(s)
	}

	for i := 0; i < lastright; i++ {
		if f.HasMissing && f.IsMissing(cases[i]) {
			continue
		}
		if f.NumData[cases[i]] <= split {
			//Left
			lastleft++
			if i != lastleft {
				swaper = cases[i]
				cases[i] = cases[lastleft]
				cases[lastleft] = swaper
				i--

			}

		} else {
			//Right
			lastright -= 1
			swaper = cases[i]
			cases[i] = cases[lastright]
			cases[lastright] = swaper
			i -= 1

		}

	}

	l = cases[:lastleft+1]
	r = cases[lastright:]
	m = cases[lastleft+1 : lastright]

	return
}

//SplitPoints returns the last left and first right index afeter reordering the cases slice froma float64 coded split.
func (f *DenseNumFeature) SplitPoints(codedSplit interface{}, cs *[]int) (int, int) {
	cases := *cs
	length := len(cases)

	lastleft := -1
	lastright := length
	swaper := 0

	//Move left cases to the start and right cases to the end so that missing cases end up
	//in between.
	split := codedSplit.(float64)

	for i := 0; i < lastright; i++ {
		if f.HasMissing && f.IsMissing(cases[i]) {
			continue
		}
		swaper = cases[i]
		if f.NumData[swaper] <= split {
			//Left
			lastleft++
			if i != lastleft {
				//swaper = cases[i]
				cases[i] = cases[lastleft]
				cases[lastleft] = swaper
				i--

			}

		} else {
			//Right
			lastright--
			//swaper = cases[i]
			cases[i] = cases[lastright]
			cases[lastright] = swaper
			i--

		}

	}
	lastleft++

	return lastleft, lastright
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
	leafSize int,
	randomSplit bool,
	allocs *BestSplitAllocs) (codedSplit interface{}, impurityDecrease float64, constant bool) {

	var nmissing, nonmissing, total int
	var nonmissingparentImp, missingimp float64
	var tosplit *[]int
	if f.HasMissing {
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
		nmissing = len(*allocs.Right)
		total = len(*cases)
		nonmissing = total - nmissing

		nonmissingparentImp = target.Impurity(allocs.NonMissing, allocs.Counter)

		if nmissing > 0 {
			missingimp = target.Impurity(allocs.Right, allocs.Counter)
		}
		tosplit = allocs.NonMissing
	} else {
		nonmissingparentImp = parentImp
		tosplit = cases
	}

	codedSplit, impurityDecrease, constant = f.BestNumSplit(target, tosplit, nonmissingparentImp, leafSize, randomSplit, allocs)

	if f.HasMissing && nmissing > 0 && impurityDecrease > minImp {
		impurityDecrease = parentImp + ((float64(nonmissing)*(impurityDecrease-nonmissingparentImp) - float64(nmissing)*missingimp) / float64(total))
	}
	return

}

/*
BestNumSplit searches over the possible splits of cases that can be made with f
and returns the one that minimizes the impurity of the target and the impurity decrease.

It expects to be provided cases for which the feature is not missing.

It searches by sorting the cases by the potential splitter and then evaluating each "gap"
between cases with non equal value as a potential split.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseNumFeature) BestNumSplit(target Target,
	cases *[]int,
	parentImp float64,
	leafSize int,
	randomSplit bool,
	allocs *BestSplitAllocs) (codedSplit interface{}, impurityDecrease float64, constant bool) {

	impurityDecrease = minImp
	var splitf float64

	ncases := len(*cases)
	if ncases >= 2*leafSize {
		sorter := allocs.Sorter

		sorter.Load(&f.NumData, cases)
		sorter.Sort()

		sorted := sorter.Cases
		sortedData := sorter.Vals

		lastsplit := 0
		var innerimp float64
		stop := (ncases - leafSize)
		constant = (sortedData[0] + constant_cutoff) >= sortedData[ncases-1]
		if constant {
			impurityDecrease = minImp
			return
		}
		lasti := leafSize - 1

		if randomSplit {
			leafSize = leafSize + rand.Intn(stop-leafSize)
			lasti = leafSize - 1
			stop = leafSize + 1

		}

		for i := leafSize; i < stop; i++ {

			//skip cases where the next sorted case has the same value as these can't be split on
			if sortedData[i] <= (sortedData[lasti] + constant_cutoff) {
				continue
				if randomSplit {
					stop++
				}
			}

			if lastsplit == 0 {
				allocs.LM = sorted[:i]
				allocs.RM = sorted[i:]
				if parentImp < 0 {
					innerimp = target.SplitImpurity(&allocs.LM, &allocs.RM, nil, allocs)
				} else {
					innerimp = parentImp
					innerimp -= target.SplitImpurity(&allocs.LM, &allocs.RM, nil, allocs)
				}

				lastsplit = i
			} else {
				allocs.LM = sorted[:i]
				allocs.RM = sorted[i:]
				allocs.MM = sorted[lastsplit:i]
				if parentImp < 0 {
					innerimp = target.UpdateSImpFromAllocs(&allocs.LM, &allocs.RM, nil, allocs, &allocs.MM)
				} else {
					innerimp = parentImp
					innerimp -= target.UpdateSImpFromAllocs(&allocs.LM, &allocs.RM, nil, allocs, &allocs.MM)
				}
				lastsplit = i
			}

			if innerimp > impurityDecrease {
				impurityDecrease = innerimp
				splitf = sortedData[lasti]
				splitf += sortedData[i]
				splitf /= 2.0
				//fmt.Println(len(sorter.Cases), sorter.Vals, allocs.LM, allocs.RM, codedSplit, impurityDecrease)

			}
			lasti = i

		}

	}
	codedSplit = splitf
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
func (target *DenseNumFeature) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {

	//This code relies on the fact that:
	// sum((xi-x_mean)^2)
	//= sum(xi^2) - n * x_mean^2
	//= sum(xi^2) - sum(xi)^2/n

	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	sum := 0.0
	sum_sqr := 0.0

	//Left impurity
	sum, sum_sqr = target.SumAndSumSquares(l)
	impurityDecrease = nl * (sum_sqr - sum*sum/nl)
	allocs.Lsum = sum
	allocs.Lsum_sqr = sum_sqr

	//Right Impurity
	sum, sum_sqr = target.SumAndSumSquares(r)
	impurityDecrease += nr * (sum_sqr - sum*sum/nr)
	allocs.Rsum = sum
	allocs.Rsum_sqr = sum_sqr

	//Missing Impurity
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		sum, sum_sqr = target.SumAndSumSquares(m)
		impurityDecrease += nm * (sum_sqr - sum*sum/nm)
		allocs.Msum = sum
		allocs.Msum_sqr = sum_sqr
	}

	impurityDecrease /= nl + nr + nm
	return
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *DenseNumFeature) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	//This code relies on the fact that:
	// sum((xi-x_mean)^2)
	//= sum(xi^2) - n * x_mean^2
	//= sum(xi^2) - sum(xi)^2/n
	//it moves the sum and sum_sqr R to L in th allocs and recalculates impurities

	MVsum, MVsum_sqr := target.SumAndSumSquares(movedRtoL)

	allocs.Lsum += MVsum
	allocs.Rsum -= MVsum
	allocs.Lsum_sqr += MVsum_sqr
	allocs.Rsum_sqr -= MVsum_sqr

	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	//Left impurity
	impurityDecrease = nl * (allocs.Lsum_sqr - allocs.Lsum*allocs.Lsum/nl)

	//Right Impurity
	impurityDecrease += nr * (allocs.Rsum_sqr - allocs.Rsum*allocs.Rsum/nr)

	//Missing Impurity
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * (allocs.Msum_sqr - allocs.Msum*allocs.Msum/nm)

	}

	impurityDecrease /= nl + nr + nm
	return
}

func (target *DenseNumFeature) SumAndSumSquares(cases *[]int) (sum float64, sum_sqr float64) {
	for _, i := range *cases {
		x := target.NumData[i]
		sum += x
		sum_sqr += x * x
	}
	return
}

//Impurity returns Gini impurity or mean squared error vs the mean for a set of cases
//depending on weather the feature is categorical or numerical
func (target *DenseNumFeature) Impurity(cases *[]int, counter *[]int) (e float64) {
	//This code relies on the fact that:
	// sum((xi-x_mean)^2)
	//= sum(xi^2) - n * x_mean^2
	//= sum(xi^2) - sum(xi)^2/n
	//the un optimized verzion would be:
	// m := target.Mean(cases)
	// e = target.Error(cases, m)

	sum, sum_sqr := target.SumAndSumSquares(cases)
	e = sum_sqr - sum*sum/float64(len(*cases))

	return

}

//Error returns the  Mean Squared error of the cases specified vs the predicted
//value. Only non missing cases are considered.
func (target *DenseNumFeature) Error(cases *[]int, predicted float64) (e float64) {
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
	counts := make(map[float64]int, 4)
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

//Span returns the lengh along the real line spaned by the specified cases
func (f *DenseNumFeature) Span(cases *[]int, counter *[]int) (span float64) {
	first := true
	min := 0.0
	max := 0.0
	val := 0.0
	for _, i := range *cases {
		if !f.Missing[i] {
			val = f.NumData[i]
			if first {
				min = val
				max = val
				continue
			}
			switch {
			case val > max:
				max = val
			case val < min:
				min = val
			}

		}

	}

	return max - min

}

//Find predicted takes the indexes of a set of cases and returns the
//predicted value. For categorical features this is a string containing the
//most common category and for numerical it is the mean of the values.
func (f *DenseNumFeature) FindPredicted(cases []int) (pred string) {
	pred = fmt.Sprintf("%v", f.Mean(&cases))
	if pred == "NaN" {
		log.Print("NaN predicted with cases ", len(cases))
	}
	return

}

//Shuffle does an inplace shuffle of the specified feature
func (f *DenseNumFeature) Shuffle() {
	capacity := len(f.Missing)
	//shuffle
	for j := 0; j < capacity; j++ {
		sourcei := j + rand.Intn(capacity-j)
		missing := f.Missing[j]
		f.Missing[j] = f.Missing[sourcei]
		f.Missing[sourcei] = missing

		data := f.NumData[j]
		f.NumData[j] = f.NumData[sourcei]
		f.NumData[sourcei] = data

	}

}

//ShuffleCases does an inplace shuffle of the specified cases
func (f *DenseNumFeature) ShuffleCases(cases *[]int) {
	capacity := len(*cases)
	//shuffle
	for j := 0; j < capacity; j++ {

		targeti := (*cases)[j]
		sourcei := (*cases)[j+rand.Intn(capacity-j)]
		missing := f.Missing[targeti]
		f.Missing[targeti] = f.Missing[sourcei]
		f.Missing[sourcei] = missing

		data := f.NumData[targeti]
		f.NumData[targeti] = f.NumData[sourcei]
		f.NumData[sourcei] = data

	}

}

/*ShuffledCopy returns a shuffled version of f for use as an artificial contrast in evaluation of
importance scores. The new feature will be named featurename:SHUFFLED*/
func (f *DenseNumFeature) ShuffledCopy() Feature {
	fake := f.Copy()
	fake.Shuffle()
	fake.(*DenseNumFeature).Name += ":SHUFFLED"
	return fake

}

/*Copy returns a copy of f.*/
func (f *DenseNumFeature) Copy() Feature {
	capacity := len(f.Missing)
	fake := &DenseNumFeature{
		nil,
		make([]bool, capacity),
		f.Name,
		false}

	copy(fake.Missing, f.Missing)

	fake.NumData = make([]float64, capacity)
	copy(fake.NumData, f.NumData)

	return fake
}

//CopyInTo copies the values and missing state from one numerical feature into another.
func (f *DenseNumFeature) CopyInTo(copyf Feature) {
	copy(copyf.(*DenseNumFeature).Missing, f.Missing)
	copy(copyf.(*DenseNumFeature).NumData, f.NumData)
}

//ImputeMissing imputes the missing values in a feature to the mean or mode of the feature.
func (f *DenseNumFeature) ImputeMissing() {
	cases := make([]int, 0, len(f.Missing))
	for i := range f.Missing {
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
	f.HasMissing = false
}
