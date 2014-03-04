package CloudForest

import ()

//BestSplitAllocs contains reusable allocations for split searching and evaluation.
//Seprate instances should be used in each go routing doing learning.
type BestSplitAllocs struct {
	Left           *[]int           //left cases for potential splits
	Right          *[]int           //right cases for potential splits
	NonMissing     *[]int           //non missing cases for potential splits
	Counter        *[]int           //class counter for counting classes in splits
	LCounter       *[]int           //left class counter iterativell sumarizing (mean) splits
	RCounter       *[]int           //right class counter iterativell sumarizing (mean) splits
	Lval           float64          //left value for iterativell sumarizing (mean) splits
	Rval           float64          //right value for iterativell sumarizing (mean) splits
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
	bsa = &BestSplitAllocs{&left,
		&right,
		&nonmissing,
		&counter,
		&lcounter,
		&rcounter,
		0.0,
		0.0,
		new(SortableFeature),
		target.(Feature).Copy().(Target)}
	return
}
