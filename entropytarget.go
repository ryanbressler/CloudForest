package CloudForest

import (
	"math"
)

/*
EntropyTarget wraps a categorical feature for use in entropy driven classification
as in Ross Quinlan's ID3 (Iterative Dichotomizer 3).
*/
type EntropyTarget struct {
	CatFeature
}

//NewEntropyTarget creates a RefretTarget and initializes EntropyTarget.Costs to the proper length.
func NewEntropyTarget(f CatFeature) *EntropyTarget {
	return &EntropyTarget{f}
}

/*
EntropyTarget.SplitImpurity is a version of Split Impurity that calls EntropyTarget.Impurity
*/
func (target *EntropyTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.Impurity(l, allocs.LCounter)
	impurityDecrease += nr * target.Impurity(r, allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.Impurity(m, allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *EntropyTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	target.MoveCountsRtoL(allocs, movedRtoL)
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.ImpFromCounts(len(*l), allocs.LCounter)
	impurityDecrease += nr * target.ImpFromCounts(len(*r), allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.ImpFromCounts(len(*m), allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

func (target *EntropyTarget) ImpFromCounts(total int, counts *[]int) (e float64) {
	p := 0.0
	for _, i := range *counts {
		if i > 0 {
			p = float64(i) / float64(total)
			e -= p * math.Log(p)
		}
	}
	return

}

//EntropyTarget.Impurity implements categorical entropy as sum(pj*log2(pj)) where pj
//is the number of cases with the j'th category over the total number of cases.
func (target *EntropyTarget) Impurity(cases *[]int, counts *[]int) (e float64) {

	total := len(*cases)
	target.CountPerCat(cases, counts)

	p := 0.0
	for _, i := range *counts {
		if i > 0 {
			p = float64(i) / float64(total)
			e -= p * math.Log(p)
		}

	}

	return

}
