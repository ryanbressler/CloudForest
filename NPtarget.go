package CloudForest

import (
	"math"
)

/*
NPTarget wraps a categorical feature for use in experimental approximate Neyman-Pearson (NP)
classification...constraints and optimization are done on percision false
positive/negative rate.

It uses an impurity measure with a soft constraint from the seccond family presented in

"Comparison and Design of Neyman-Pearson Classiﬁers"
Clayton Scott,  October 2005

http://www.stat.rice.edu/~cscott/pubs/npdesign.pdf


N(f) = κ max((R0(f) − α), 0) + R1(f)

Where f is the classifer, R0 is the flase positive rate R1 is the false negative rate,
α is the false positive constraint and k controls the cost of violating
this constraint and β is a constant we can ignore as it subtracts out in diffrences.

The vote assigned to each leaf node is a corrected mode where the count of the
positive/constrained label is corrected by 1/α. Without this modification constraints
> .5 won't work since nodes with that many negatives false positives won't vote positive.
*/
type NPTarget struct {
	CatFeature
	Posi  int
	Alpha float64
	Kappa float64
}

//NewNPTarget wraps a Categorical Feature for NP Classification. It accepts
//a string representing the contstrained label and floats Alpha and Kappa
//representing the constraint and constraint weight.
func NewNPTarget(f CatFeature, Pos string, Alpha, Kappa float64) *NPTarget {
	return &NPTarget{f, f.CatToNum(Pos), Alpha, Kappa}
}

/*
SplitImpurity is a version of Split Impurity that calls NPTarget.Impurity
*/
func (target *NPTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
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

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l
//to avoid recalulatign the entire split impurity.
func (target *NPTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	var cat, i int
	lcounter := *allocs.LCounter
	rcounter := *allocs.RCounter
	for _, i = range *movedRtoL {

		//most expensive statement:
		cat = target.Geti(i)
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
		nm = float64(len(*m))
		impurityDecrease += nm * target.ImpFromCounts(len(*m), allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//FindPredicted does a mode calulation with the count of the positive/constrained
//class corrected.
func (target *NPTarget) FindPredicted(cases []int) (pred string) {

	mi := 0
	mc := 0.0
	counts := make([]int, target.NCats())

	target.CountPerCat(&cases, &counts)

	for cat, count := range counts {
		cc := float64(count)
		if cat == target.Posi {
			cc /= target.Alpha
		}
		if cc > mc {
			mi = cat
			mc = cc
		}
	}

	return target.NumToCat(mi)

}

//ImpFromCounts recalculates gini impurity from class counts for us in intertive updates.
func (target *NPTarget) ImpFromCounts(t int, counter *[]int) (e float64) {

	var totalpos, totalneg, mi int

	mc := 0.0

	for cat, count := range *counter {
		cc := float64(count)
		if cat == target.Posi {
			totalpos += count
			cc /= target.Alpha
		} else {
			totalneg += count
		}

		if cc > mc {
			mi = cat
			mc = cc
		}

	}

	if target.Posi == mi {
		//False positive constraint
		e = target.Kappa * math.Max(float64(totalneg)/float64(t)-target.Alpha, 0)
	} else {
		//False negative rate
		e = float64(totalpos) / float64(t)
	}

	return

}

//NPTarget.Impurity implements an impurity that minimizes false negatives subject
//to a soft constrain on fale positives.
func (target *NPTarget) Impurity(cases *[]int, counter *[]int) (e float64) {

	target.CountPerCat(cases, counter)
	t := len(*cases)
	e = target.ImpFromCounts(t, counter)

	return

}
