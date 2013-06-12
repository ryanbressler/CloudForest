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
func (target *EntropyTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//EntropyTarget.Impurity implements categorical entropy as sum(pj*log2(pj)) where pj
//is the number of cases with the j'th category over the total number of cases.
func (target *EntropyTarget) Impurity(cases *[]int, counts *[]int) (e float64) {
	total := 0
	counter := *counts
	for i, _ := range counter {
		counter[i] = 0
	}
	for _, i := range *cases {
		if !target.IsMissing(i) {
			counter[target.Geti(i)] += 1
			total += 1
		}
	}
	e = 0.0
	p := 0.0
	for _, v := range counter {
		p = float64(v) / float64(total)
		e -= p * math.Log2(p)
	}

	return

}
