package CloudForest

import (
	"math"
)

/*
NPTarget wraps a categorical feature for use approximate NP classification.
It uses an impurity measure derived from the seccond family presented in

"Comparison and Design of Neyman-Pearson Classiﬁers"
Clayton Scott,  October 2005

http://www.stat.rice.edu/~cscott/pubs/npdesign.pdf


N(f) = κ max((R0(f) − α), 0) + R1(f) − β.

Where α is the false positive constraint and k controls the cost of violating
this constraint and β is a constant we can ignore.
*/
type NPTarget struct {
	CatFeature
	Posi  int
	Alpha float64
	Kappa float64
}

//NewNPTarget creates a RefretTarget and initializes NPTarget.Costs to the proper length.
func NewNPTarget(f CatFeature, Pos string, Alpha, Kappa float64) *NPTarget {
	return &NPTarget{f, f.CatToNum(Pos), Alpha, Kappa}
}

/*
NPTarget.SplitImpurity is a version of Split Impurity that calls NPTarget.Impurity
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

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *NPTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	return target.SplitImpurity(l, r, m, allocs)
}

//NPTarget.Impurity implements a simple regret function that finds the average cost of
//a set using the misclassification costs in NPTarget.Costs.
func (target *NPTarget) Impurity(cases *[]int, counter *[]int) (e float64) {

	var t, totalpos, totalneg int
	for _, c := range *cases {
		if target.IsMissing(c) == false {
			t++
			cat := target.Geti(c)
			if cat == target.Posi {
				totalpos++
			} else {
				totalneg++
			}
		}

	}

	if target.Posi == target.Modei(cases) {
		//False positive constraint
		e = target.Kappa * math.Max(float64(totalneg)/float64(t)-target.Alpha, 0)
	} else {
		//False negative rate
		e = float64(totalpos) / float64(t)
	}

	// e = float64(totalneg) * target.Kappa * math.Max(float64(totalneg)/float64(t)-target.Alpha, 0)
	// e += float64(totalpos) * float64(totalpos) / float64(t)
	// e /= float64(t)

	return
}
