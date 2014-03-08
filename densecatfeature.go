package CloudForest

import (
	"math/big"
	"math/rand"
	"strings"
)

/*DenseCatFeature is a structure representing a single feature in a feature matrix.
It contains:
An embedded CatMap (may only be instantiated for cat data)
	NumData   : A slice of floates used for numerical data and nil otherwise
	CatData   : A slice of ints for categorical data and nil otherwise
	Missing   : A slice of bools indicating missing values. Measure this for length.
	Numerical : is the feature numerical
	Name      : the name of the feature*/
type DenseCatFeature struct {
	*CatMap
	CatData      []int
	Missing      []bool
	Name         string
	RandomSearch bool
	HasMissing   bool
}

//Append will parse and append a single value to the end of the feature. It is generally only used
//during data parseing.
func (f *DenseCatFeature) Append(v string) {
	norm := strings.ToLower(v)
	if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {

		f.CatData = append(f.CatData, 0)
		f.Missing = append(f.Missing, true)
		f.HasMissing = true
		return
	}
	f.CatData = append(f.CatData, f.CatToNum(v))
	f.Missing = append(f.Missing, false)
}

func (f *DenseCatFeature) Length() int {
	return len(f.Missing)
}

func (f *DenseCatFeature) GetName() string {
	return f.Name
}

func (f *DenseCatFeature) IsMissing(i int) bool {
	return f.Missing[i]
}

func (f *DenseCatFeature) MissingVals() bool {
	return f.HasMissing
}

func (f *DenseCatFeature) PutMissing(i int) {
	f.Missing[i] = true
	f.HasMissing = true
}

func (f *DenseCatFeature) Geti(i int) int {
	return f.CatData[i]
}

func (f *DenseCatFeature) Puti(i int, v int) {
	f.CatData[i] = v
	f.Missing[i] = false
}

func (f *DenseCatFeature) GetStr(i int) string {
	if f.Missing[i] {
		return "NA"
	}
	return f.Back[f.CatData[i]]
}

func (f *DenseCatFeature) PutStr(i int, v string) {
	norm := strings.ToLower(v)
	if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {
		f.Missing[i] = true
		f.HasMissing = true
	}
	vi := f.CatToNum(v)
	f.CatData[i] = vi
	f.Missing[i] = false
}

func (f *DenseCatFeature) GoesLeft(i int, splitter *Splitter) bool {
	return splitter.Left[f.Back[f.CatData[i]]]
}

/*
BestSplit finds the best split of the features that can be achieved using
the specified target and cases. It returns a Splitter and the decrease in impurity.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestSplit(target Target,
	cases *[]int,
	parentImp float64,
	leafSize int,
	allocs *BestSplitAllocs) (codedSplit interface{}, impurityDecrease float64) {

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

	//TODO: reverse this list, common cases first and maybe make it a switch statement
	nCats := f.NCats()
	if f.RandomSearch == false && nCats > maxNonBigCats {
		codedSplit, impurityDecrease = f.BestCatSplitIterBig(target, tosplit, nonmissingparentImp, leafSize, allocs)
	} else if f.RandomSearch == false && nCats > maxExhaustiveCats {
		codedSplit, impurityDecrease = f.BestCatSplitIter(target, tosplit, nonmissingparentImp, leafSize, allocs)
	} else if nCats > maxNonBigCats {
		codedSplit, impurityDecrease = f.BestCatSplitBig(target, tosplit, nonmissingparentImp, maxNonRandomExahustive, leafSize, allocs)
	} else if nCats == 2 {
		codedSplit, impurityDecrease = f.BestBinSplit(target, tosplit, nonmissingparentImp, maxNonRandomExahustive, leafSize, allocs)
	} else {
		codedSplit, impurityDecrease = f.BestCatSplit(target, tosplit, nonmissingparentImp, maxNonRandomExahustive, leafSize, allocs)
	}

	if f.HasMissing && nmissing > 0 && impurityDecrease > minImp {
		impurityDecrease = parentImp + ((float64(nonmissing)*(impurityDecrease-nonmissingparentImp) - float64(nmissing)*missingimp) / float64(total))
	}
	return

}

func (f *DenseCatFeature) Split(codedSplit interface{}, cases []int) (l []int, r []int, m []int) {
	length := len(cases)

	lastleft := -1
	lastright := length
	swaper := 0

	var GoesLeft func(int) bool

	switch codedSplit.(type) {
	case int:
		cat := codedSplit.(int)
		// doesn't account for non slitting case cat = 3
		// or left vs right simitry which makes tests fail
		// if f.NCats() == 2 {
		// 	GoesLeft = func(i int) bool {
		// 		return f.CatData[i] != cat
		// 	}
		// } else {
		GoesLeft = func(i int) bool {
			return 0 != (cat & (1 << uint(f.CatData[i])))
		}
		// }
	case *big.Int:
		bigCat := codedSplit.(*big.Int)
		GoesLeft = func(i int) bool {
			return 0 != bigCat.Bit(f.CatData[i])
		}

	}

	//Move left cases to the start and right cases to the end so that missing cases end up
	//in between.

	for i := 0; i < lastright; i++ {
		if f.HasMissing && f.IsMissing(cases[i]) {
			continue
		}
		if GoesLeft(cases[i]) { //Left
			lastleft++
			if i != lastleft {

				swaper = cases[i]
				cases[i] = cases[lastleft]
				cases[lastleft] = swaper
				i--
			}
		} else { //Right
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

//Decode split builds a splitter from the numeric values returned by BestNumSplit or
//BestCatSplit. Numeric splitters are decoded to send values <= num left. Categorical
//splitters are decoded to send categorical values for which the bit in cat is 1 left.
func (f *DenseCatFeature) DecodeSplit(codedSplit interface{}) (s *Splitter) {

	nCats := f.NCats()
	s = &Splitter{f.Name, false, 0.0, make(map[string]bool, nCats)}

	switch codedSplit.(type) {
	case int:
		cat := codedSplit.(int)
		for j := 0; j < nCats; j++ {

			if 0 != (cat & (1 << uint(j))) {
				s.Left[f.Back[j]] = true
			}

		}
	case *big.Int:
		bigCat := codedSplit.(*big.Int)
		for j := 0; j < nCats; j++ {
			if 0 != bigCat.Bit(j) {
				s.Left[f.Back[j]] = true
			}
		}

	}

	return
}

/*BestCatSplitIterBig performs an iterative search to find the split that minimizes impurity
in the specified target. It expects to be provided for cases fir which the feature is not missing.

Searching is implemented via bitwise on integers for speed but will currently only work
when there are less categories then the number of bits in an int.

The best split is returned as an int for which the bits corresponding to categories that should
be sent left has been flipped. This can be decoded into a splitter using DecodeSplit on the
training feature and should not be applied to testing data without doing so as the order of
categories may have changed.


allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestCatSplitIterBig(target Target, cases *[]int, parentImp float64, leafSize int, allocs *BestSplitAllocs) (bestSplit *big.Int, impurityDecrease float64) {

	left := *allocs.Left
	right := *allocs.Right
	/*
		This is an iterative search for the best combinations of categories.
	*/

	nCats := f.NCats()
	cat := 0

	//overall running best impurity and split
	impurityDecrease = minImp
	bestSplit = big.NewInt(0)

	//running best with n categories
	innerImp := minImp
	innerSplit := big.NewInt(0)

	//values for the current proposed n+1 category
	nextImp := minImp
	nextSplit := big.NewInt(0)

	//iteratively build a combination of categories until they
	//stop getting better
	for j := 0; j < nCats; j++ {

		innerImp = impurityDecrease
		innerSplit.SetInt64(0)
		//find the best additional category
		for i := 0; i < nCats; i++ {

			if bestSplit.Bit(i) != 0 {
				continue
			}

			left = left[0:0]
			right = right[0:0]

			nextSplit.SetBit(bestSplit, i, 1)

			for _, c := range *cases {

				cat = f.CatData[c]
				if 0 != nextSplit.Bit(cat) {
					left = append(left, c)
				} else {
					right = append(right, c)
				}

			}

			//skip cases where the split didn't do any splitting
			if len(left) < leafSize || len(right) < leafSize {
				continue
			}

			nextImp = parentImp - target.SplitImpurity(&left, &right, nil, allocs)

			if nextImp > innerImp {
				innerSplit.Set(nextSplit)
				innerImp = nextImp

			}

		}
		if innerImp > impurityDecrease {
			bestSplit.Set(innerSplit)
			impurityDecrease = innerImp

		} else {
			break
		}

	}

	return
}

/*
BestCatSplitIter performs an iterative search to find the split that minimizes impurity
in the specified target. It expects to be provided for cases fir which the feature is not missing.

Searching is implemented via bitwise ops on ints (32 bit) for speed but will currently only work
when there are <31 categories. Use BigInterBestCatSplit above that.

The best split is returned as an int for which the bits corresponding to categories that should
be sent left has been flipped. This can be decoded into a splitter using DecodeSplit on the
training feature and should not be applied to testing data without doing so as the order of
categories may have changed.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestCatSplitIter(target Target, cases *[]int, parentImp float64, leafSize int, allocs *BestSplitAllocs) (bestSplit int, impurityDecrease float64) {

	left := *allocs.Left
	right := *allocs.Right
	/*
		This is an iterative search for the best combinations of categories.
	*/

	nCats := f.NCats()
	cat := 0

	//overall running best impurity and split
	impurityDecrease = minImp
	bestSplit = 0

	//running best with n categories
	innerImp := minImp
	innerSplit := 0

	//values for the current proposed n+1 category
	nextImp := minImp
	nextSplit := 0

	//iteratively build a combination of categories until they
	//stop getting better
	for j := 0; j < nCats; j++ {

		innerImp = impurityDecrease
		innerSplit = 0
		//find the best additional category
		for i := 0; i < nCats; i++ {

			if 0 != (bestSplit & (1 << uint(i))) {
				continue
			}

			left = left[0:0]
			right = right[0:0]

			nextSplit = bestSplit | 1<<uint(i)

			for _, c := range *cases {

				cat = f.CatData[c]
				if 0 != (nextSplit & (1 << uint(cat))) {
					left = append(left, c)
				} else {
					right = append(right, c)
				}

			}

			//skip cases where the split didn't do any splitting
			if len(left) < leafSize || len(right) < leafSize {
				continue
			}

			nextImp = parentImp - target.SplitImpurity(&left, &right, nil, allocs)

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
BestCatSplit performs an exhaustive search for the split that minimizes impurity
in the specified target for categorical features with less then 31 categories.
It expects to be provided for cases fir which the feature is not missing.

This implementation follows Brieman's implementation and the R/Matlab implementations
based on it use exhaustive search for when there are less than 25/10 categories
and random splits above that.

Searching is implemented via bitwise operations vs an incrementing or random int (32 bit) for speed
but will currently only work when there are less then 31 categories. Use one of the Big functions
above that.

The best split is returned as an int for which the bits corresponding to categories that should
be sent left has been flipped. This can be decoded into a splitter using DecodeSplit on the
training feature and should not be applied to testing data without doing so as the order of
categories may have changed.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestCatSplit(target Target,
	cases *[]int,
	parentImp float64,
	maxEx int,
	leafSize int,
	allocs *BestSplitAllocs) (bestSplit int, impurityDecrease float64) {

	impurityDecrease = minImp
	left := *allocs.Left
	right := *allocs.Right
	/*

		Exhaustive search of combinations of categories is carried out by iterating an Int and using
		the bits to define which categories go to the left of the split.

	*/
	nCats := f.NCats()

	useExhaustive := nCats <= maxEx
	nPartitions := 1
	if useExhaustive {
		//2**(nCats-2) is the number of valid partitions (collapsing symmetric partitions)
		nPartitions = (2 << uint(nCats-2))
	} else {
		//if more then the max we will loop max times and generate random combinations
		nPartitions = (2 << uint(maxEx-2))
	}
	bestSplit = 0
	bits := 0
	innerimp := 0.0
	//start at 1 to ignore the set with all on one side
	for i := 1; i < nPartitions; i++ {

		bits = i
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

			j = f.CatData[c]
			if 0 != (bits & (1 << uint(j))) {
				left = append(left, c)
			} else {
				right = append(right, c)
			}

		}

		//skip cases where the split didn't do any splitting
		if len(left) < leafSize || len(right) < leafSize {
			continue
		}

		innerimp = parentImp - target.SplitImpurity(&left, &right, nil, allocs)

		if innerimp > impurityDecrease {
			bestSplit = bits
			impurityDecrease = innerimp

		}

	}

	return
}

/*
BestBinSplit performs an exhaustive search for the split that minimizes impurity
in the specified target for categorical features with 2 categories.
It expects to be provided for cases fir which the feature is not missing.

This implementation follows Brieman's implementation and the R/Matlab implementations
based on it use exhaustive search for when there are less than 25/10 categories
and random splits above that.

Searching is implemented via bitwise operations vs an incrementing or random int (32 bit) for speed
but will currently only work when there are less then 31 categories. Use one of the Big functions
above that.

The best split is returned as an int for which the bits corresponding to categories that should
be sent left has been flipped. This can be decoded into a splitter using DecodeSplit on the
training feature and should not be applied to testing data without doing so as the order of
categories may have changed.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestBinSplit(target Target,
	cases *[]int,
	parentImp float64,
	maxEx int,
	leafSize int,
	a *BestSplitAllocs) (bestSplit int, impurityDecrease float64) {

	a.L = a.L[0:0]
	a.R = a.R[0:0]

	catdata := f.CatData
	for _, c := range *cases {

		if catdata[c] == 1 {
			a.R = append(a.R, c)
		} else {
			a.L = append(a.L, c)
		}

	}

	//skip cases where the split didn't do any splitting
	if len(a.L) < leafSize || len(a.R) < leafSize {
		return
	}

	impurityDecrease = parentImp - target.SplitImpurity(&a.L, &a.R, nil, a)

	bestSplit = 1

	return
}

/*BestCatSplitBig performs a random/exhaustive search to find the split that minimizes impurity
in the specified target. It expects to be provided for cases fir which the feature is not missing.

Searching is implemented via bitwise on Big.Ints to handle large n categorical features but BestCatSplit
should be used for n <31.

The best split is returned as a BigInt for which the bits corresponding to categories that should
be sent left has been flipped. This can be decoded into a splitter using DecodeSplit on the
training feature and should not be applied to testing data without doing so as the order of
categories may have changed.

allocs contains pointers to reusable structures for use while searching for the best split and should
be initialized to the proper size with NewBestSplitAlocs.
*/
func (f *DenseCatFeature) BestCatSplitBig(target Target, cases *[]int, parentImp float64, maxEx int, leafSize int, allocs *BestSplitAllocs) (bestSplit *big.Int, impurityDecrease float64) {

	left := *allocs.Left
	right := *allocs.Right

	nCats := f.NCats()

	//overall running best impurity and split
	impurityDecrease = minImp
	bestSplit = big.NewInt(0)

	//running best with n categories
	innerImp := minImp

	bits := big.NewInt(1)

	var randgn *rand.Rand
	var maxPart *big.Int
	useExhaustive := nCats <= maxEx
	nPartitions := big.NewInt(2)
	if useExhaustive {
		//2**(nCats-2) is the number of valid partitions (collapsing symmetric partitions)
		nPartitions.Lsh(nPartitions, uint(nCats-2))
	} else {
		//if more then the max we will loop max times and generate random combinations
		nPartitions.Lsh(nPartitions, uint(maxEx-2))
		maxPart = big.NewInt(2)
		maxPart.Lsh(maxPart, uint(nCats-2))
		randgn = rand.New(rand.NewSource(0))
	}

	//iteratively build a combination of categories until they
	//stop getting better
	for i := big.NewInt(1); i.Cmp(nPartitions) == -1; i.Add(i, big.NewInt(1)) {

		bits.Set(i)
		if !useExhaustive {
			//generate random partition
			bits.Rand(randgn, maxPart)
		}

		//check the value of the j'th bit of i and
		//send j left or right
		left = left[0:0]
		right = right[0:0]
		j := 0
		for _, c := range *cases {

			j = f.CatData[c]
			if 0 != bits.Bit(j) {
				left = append(left, c)
			} else {
				right = append(right, c)
			}

		}

		//skip cases where the split didn't do any splitting
		if len(left) < leafSize || len(right) < leafSize {
			continue
		}

		innerImp = parentImp - target.SplitImpurity(&left, &right, nil, allocs)

		if innerImp > impurityDecrease {
			bestSplit.Set(bits)
			impurityDecrease = innerImp

		}

	}

	return
}

/*
FilterMissing loops over the cases and appends them into filtered.
For most use cases filtered should have zero length before you begin as it is not reset
internally
*/
func (f *DenseCatFeature) FilterMissing(cases *[]int, filtered *[]int) {
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
func (target *DenseCatFeature) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
	// l := *left
	// r := *right
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.GiniWithoutAlocate(l, allocs.LCounter)
	impurityDecrease += nr * target.GiniWithoutAlocate(r, allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.GiniWithoutAlocate(m, allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

func (target *DenseCatFeature) MoveCountsRtoL(allocs *BestSplitAllocs, movedRtoL *[]int) {
	var cat, i int
	catdata := target.CatData
	lcounter := *allocs.LCounter
	rcounter := *allocs.RCounter
	for _, i = range *movedRtoL {

		//most expensive statement:
		cat = catdata[i]
		lcounter[cat]++
		rcounter[cat]--

	}
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *DenseCatFeature) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	var cat, i int
	catdata := target.CatData
	lcounter := *allocs.LCounter
	rcounter := *allocs.RCounter
	for _, i = range *movedRtoL {

		//most expensive statement:
		cat = catdata[i]
		lcounter[cat]++
		rcounter[cat]--
		//counter[target.Geti(i)]++

	}
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.ImpFromCounts(len(*l), allocs.LCounter)
	impurityDecrease += nr * target.ImpFromCounts(len(*r), allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm := float64(len(*m))
		impurityDecrease += nm * target.ImpFromCounts(len(*m), allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//Impurity returns Gini impurity or mean squared error vs the mean for a set of cases
//depending on weather the feature is categorical or numerical
func (target *DenseCatFeature) Impurity(cases *[]int, counter *[]int) (e float64) {

	e = target.GiniWithoutAlocate(cases, counter)

	return

}

//Gini returns the gini impurity for the specified cases in the feature
//gini impurity is calculated as 1 - Sum(fi^2) where fi is the fraction
//of cases in the ith catagory.
func (target *DenseCatFeature) Gini(cases *[]int) (e float64) {
	counter := make([]int, target.NCats())
	e = target.GiniWithoutAlocate(cases, &counter)
	return
}

//CountPerCat puts per catagory counts in the supplied counter. It is designed for use in
//a target and doesn't check for missing values.
func (target *DenseCatFeature) CountPerCat(cases *[]int, counts *[]int) {
	//this function is a hot spot
	//cs := *cases
	counter := *counts
	//fastest to derfrence this outside of loop?
	catdata := target.CatData
	i := 0
	for i, _ = range counter {
		counter[i] = 0
	}

	for _, i = range *cases {
		//most expensive statement:
		counter[catdata[i]]++
		//counter[target.Geti(i)]++
	}

}

/*
giniWithoutAlocate calculates gini impurity using the supplied counter which must
be a slice with length equal to the number of cases. This allows you to reduce allocations
but the counter will also contain per category counts.
*/
func (target *DenseCatFeature) GiniWithoutAlocate(cases *[]int, counts *[]int) (e float64) {

	total := len(*cases)
	target.CountPerCat(cases, counts)
	//fastest way to set e to 1.0?
	e++
	t := float64(total * total)
	for _, i := range *counts {
		e -= float64(i*i) / t
	}
	return
}

func (target *DenseCatFeature) ImpFromCounts(total int, counts *[]int) (e float64) {
	e++
	t := float64(total * total)
	for _, i := range *counts {
		e -= float64(i*i) / t
	}
	return

}

//DistinctCats counts the number of distincts cats present in the specified cases.
func (target *DenseCatFeature) DistinctCats(cases *[]int, counts *[]int) (total int) {
	total = 0
	counter := *counts
	for i, _ := range counter {
		counter[i] = 0
	}
	for _, i := range *cases {
		if !target.Missing[i] {
			counter[target.CatData[i]] = 1
		}
	}
	for _, v := range counter {
		if v > 0 {
			total += 1
		}
	}
	return
}

//Mode returns the mode category feature for the cases specified
func (f *DenseCatFeature) Mode(cases *[]int) (m string) {
	m = f.Back[f.Modei(cases)]
	return

}

//Mode returns the mode category feature for the cases specified
func (f *DenseCatFeature) Modei(cases *[]int) (m int) {
	counts := make([]int, f.NCats())
	for _, i := range *cases {
		if !f.Missing[i] {
			counts[f.CatData[i]] += 1
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
func (f *DenseCatFeature) FindPredicted(cases []int) (pred string) {

	pred = f.Mode(&cases)

	return

}

//Shuffle does an inflace shuffle of the specified feature
func (f *DenseCatFeature) Shuffle() {
	capacity := len(f.Missing)
	//shuffle
	for j := 0; j < capacity; j++ {
		sourcei := j + rand.Intn(capacity-j)
		missing := f.Missing[j]
		f.Missing[j] = f.Missing[sourcei]
		f.Missing[sourcei] = missing

		data := f.CatData[j]
		f.CatData[j] = f.CatData[sourcei]
		f.CatData[sourcei] = data

	}

}

//ShuffleCases does an inplace shuffle of the specified cases
func (f *DenseCatFeature) ShuffleCases(cases *[]int) {
	capacity := len(*cases)
	//shuffle
	for j := 0; j < capacity; j++ {

		targeti := (*cases)[j]
		sourcei := (*cases)[j+rand.Intn(capacity-j)]
		missing := f.Missing[targeti]
		f.Missing[targeti] = f.Missing[sourcei]
		f.Missing[sourcei] = missing

		data := f.CatData[targeti]
		f.CatData[targeti] = f.CatData[sourcei]
		f.CatData[sourcei] = data

	}

}

/*ShuffledCopy returns a shuffled version of f for use as an artificial contrast in evaluation of
importance scores. The new feature will be named featurename:SHUFFLED*/
func (f *DenseCatFeature) ShuffledCopy() Feature {
	fake := f.Copy()
	fake.Shuffle()
	fake.(*DenseCatFeature).Name += ":SHUFFLED"
	return fake

}

/*Copy returns a copy of f.*/
func (f *DenseCatFeature) Copy() Feature {
	capacity := len(f.Missing)
	fake := &DenseCatFeature{
		&CatMap{f.Map,
			f.Back},
		nil,
		make([]bool, capacity),
		f.Name,
		f.RandomSearch,
		f.HasMissing}

	copy(fake.Missing, f.Missing)

	fake.CatData = make([]int, capacity)
	copy(fake.CatData, f.CatData)

	return fake

}

func (f *DenseCatFeature) CopyInTo(copyf Feature) {
	copy(copyf.(*DenseCatFeature).Missing, f.Missing)
	copy(copyf.(*DenseCatFeature).CatData, f.CatData)
}

//ImputeMissing imputes the missing values in a feature to the mean or mode of the feature.
func (f *DenseCatFeature) ImputeMissing() {
	cases := make([]int, 0, len(f.Missing))
	for i, _ := range f.Missing {
		cases = append(cases, i)
	}

	mode := f.Modei(&cases)

	for i, m := range f.Missing {
		if m {

			f.CatData[i] = mode

			f.Missing[i] = false

		}
	}
	f.HasMissing = false
}
