package CloudForest

import (
	"fmt"
	"math"
)

/*
DEntropyTarget wraps a categorical feature for use in entropy driven classification
as in Ross Quinlan's ID3 (Iterative Dichotomizer 3) with a the entropy modified to use
"disutility entropy"

I = - k Sum ri * pi * log(pi)

*/
type DEntropyTarget struct {
	CatFeature
	Costs []float64
}

//NewDEntropyTarget creates a RefretTarget and initializes DEntropyTarget.Costs to the proper length.
func NewDEntropyTarget(f CatFeature) *DEntropyTarget {
	return &DEntropyTarget{f, make([]float64, f.NCats())}
}

/*NewDEntropyTarget.SetCosts puts costs in a map[string]float64 by feature name into the proper
entries in NewDEntropyTarget.Costs.*/
func (target *DEntropyTarget) SetCosts(costmap map[string]float64) {
	for i := 0; i < target.NCats(); i++ {
		c := target.NumToCat(i)
		target.Costs[i] = costmap[c]
	}
}

/*
DEntropyTarget.SplitImpurity is a version of Split Impurity that calls DEntropyTarget.Impurity
*/
func (target *DEntropyTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
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
func (target *DEntropyTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
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

func (target *DEntropyTarget) ImpFromCounts(total int, counts *[]int) (e float64) {
	p := 0.0
	for c, i := range *counts {
		if i > 0 {
			p = float64(i) / float64(total)
			e -= target.Costs[c] * p * math.Log(p)
		}
	}
	return

}

func (target *DEntropyTarget) FindPredicted(cases []int) (pred string) {
	prob_true := 0.0
	t := target.CatToNum("True")
	weightedvoted := true
	if weightedvoted {
		count := 0.0
		total := 0.0
		for _, i := range cases {
			ti := target.Geti(i)
			cost := target.Costs[ti]
			if ti == t {
				count += cost
			}
			total += cost

		}
		prob_true = count / total

	} else {
		count := 0
		for _, i := range cases {
			if target.Geti(i) == t {
				count++
			}

		}
		prob_true = float64(count) / float64(len(cases))
	}
	return fmt.Sprintf("%v", prob_true)
}

//DEntropyTarget.Impurity implements categorical entropy as sum(pj*log2(pj)) where pj
//is the number of cases with the j'th category over the total number of cases.
func (target *DEntropyTarget) Impurity(cases *[]int, counts *[]int) (e float64) {

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
