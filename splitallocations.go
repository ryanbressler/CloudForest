package CloudForest

import ()

//BestSplitAllocs contains reusable allocations for split searching.
type BestSplitAllocs struct {
	Left       *[]int
	Right      *[]int
	NonMissing *[]int
	Counter    *[]int
	Sorter     *SortableFeature
}

//NewBestSplitAllocs initializes all of the reusable allocations for split
//searching to the appropriate size. nTotalCases should be number of total
//cases in the feature matrix being analyzed.
func NewBestSplitAllocs(nTotalCases int, target Target) (bsa *BestSplitAllocs) {
	left := make([]int, 0, nTotalCases)
	right := make([]int, 0, nTotalCases)
	nonmissing := make([]int, 0, nTotalCases)
	counter := make([]int, target.NCats())
	bsa = &BestSplitAllocs{&left,
		&right,
		&nonmissing,
		&counter,
		new(SortableFeature)}
	return
}
