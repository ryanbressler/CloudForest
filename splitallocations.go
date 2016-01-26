package CloudForest

//BestSplitAllocs contains reusable allocations for split searching and evaluation.
//Seprate instances should be used in each go routing doing learning.
type BestSplitAllocs struct {
	L              []int //Allocated to size
	R              []int
	M              []int
	LM             []int //Used to point at other array
	RM             []int
	MM             []int
	Left           *[]int  //left cases for potential splits
	Right          *[]int  //right cases for potential splits
	NonMissing     *[]int  //non missing cases for potential splits
	Counter        *[]int  //class counter for counting classes in splits used alone of for missing
	LCounter       *[]int  //left class counter sumarizing (mean) splits
	RCounter       *[]int  //right class counter sumarizing (mean) splits
	Lsum           float64 //left value for sumarizing splits
	Rsum           float64 //right value for sumarizing  splits
	Msum           float64 //missing value for sumarizing splits
	Lsum_sqr       float64 //left value for sumarizing splits
	Rsum_sqr       float64 //right value for sumarizing  splits
	Msum_sqr       float64 //missing value for sumarizing splits
	CatVals        []int
	SortVals       []float64
	Sorter         *SortableFeature //for learning from numerical features
	ContrastTarget Target
}

//NewBestSplitAllocs initializes all of the reusable allocations for split
//searching to the appropriate size. nTotalCases should be number of total
//cases in the feature matrix being analyzed.
func NewBestSplitAllocs(nTotalCases int, target Target) (bsa *BestSplitAllocs) {
	left := make([]int, 0, nTotalCases)
	right := make([]int, 0, nTotalCases)
	nonmissing := make([]int, 0, nTotalCases)
	counter := make([]int, target.NCats())
	lcounter := make([]int, target.NCats())
	rcounter := make([]int, target.NCats())
	bsa = &BestSplitAllocs{make([]int, 0, nTotalCases),
		make([]int, 0, nTotalCases),
		make([]int, 0, nTotalCases),
		nil,
		nil,
		nil,
		// make([]int, 0, nTotalCases),
		// make([]int, nTotalCases, nTotalCases),
		&left,
		&right,
		&nonmissing,
		&counter,
		&lcounter,
		&rcounter,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		make([]int, nTotalCases, nTotalCases),
		make([]float64, nTotalCases, nTotalCases),
		&SortableFeature{make([]float64, nTotalCases, nTotalCases),
			nil},
		target.(Feature).Copy().(Target)}
	return
}
