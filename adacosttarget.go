package CloudForest

import (
	"math"
)

/*
AdaCostTarget wraps a numerical feature as a target for us in Cost Sensitive Adaptive Boosting (AdaC2.M1)

"Boosting for Learning Multiple Classes with Imbalanced Class Distribution"
Yanmin Sun, Mohamed S. Kamel and Yang Wang

See equations in slides here:
http://people.ee.duke.edu/~lcarin/Minhua4.18.08.pdf

*/
type AdaCostTarget struct {
	CatFeature
	Weights []float64
	Costs   []float64
}

/*
NewAdaCostTarget creates a categorical adaptive boosting target and initializes its weights.
*/
func NewAdaCostTarget(f CatFeature) (abt *AdaCostTarget) {
	nCases := f.Length()
	abt = &AdaCostTarget{f, make([]float64, nCases), make([]float64, f.NCats())}
	for i := range abt.Weights {
		abt.Weights[i] = 1 / float64(nCases)
	}
	return
}

/*RegretTarget.SetCosts puts costs in a map[string]float64 by feature name into the proper
entries in RegretTarget.Costs.*/
func (target *AdaCostTarget) SetCosts(costmap map[string]float64) {
	for i := 0; i < target.NCats(); i++ {
		c := target.NumToCat(i)
		target.Costs[i] = costmap[c]
	}
}

/*
SplitImpurity is an AdaCosting version of SplitImpurity.
*/
func (target *AdaCostTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
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
func (target *AdaCostTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
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

	impurityDecrease = nl * target.ImpFromCounts(l, allocs.LCounter)
	impurityDecrease += nr * target.ImpFromCounts(r, allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.ImpFromCounts(m, allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//Impurity is an AdaCosting that uses the weights specified in weights.
func (target *AdaCostTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	e = 0.0
	//m := target.Modei(cases)

	target.CountPerCat(cases, counter)
	e = target.ImpFromCounts(cases, counter)

	return
}

//ImpFromCounts recalculates gini impurity from class counts for us in intertive updates.
func (target *AdaCostTarget) ImpFromCounts(cases *[]int, counter *[]int) (e float64) {

	var m, mc int

	for i, c := range *counter {
		if c > mc {
			m = i
			mc = c
		}
	}

	for _, c := range *cases {

		cat := target.Geti(c)
		if cat != m {
			e += target.Weights[c] * target.Costs[cat]
		}

	}

	return

}

//Boost performs categorical adaptive boosting using the specified partition and
//returns the weight that tree that generated the partition should be given.
func (t *AdaCostTarget) Boost(leaves *[][]int) (weight float64) {
	weight = 0.0
	counter := make([]int, t.NCats())
	for _, cases := range *leaves {
		weight += t.Impurity(&cases, &counter)
	}
	if weight >= .5 {
		return 0.0
	}
	weight = .5 * math.Log((1-weight)/weight)

	for _, cases := range *leaves {
		t.CountPerCat(&cases, &counter)

		var m, mc int
		for i, c := range counter {
			if c > mc {
				m = i
				mc = c
			}
		}

		for _, c := range cases {
			if t.IsMissing(c) == false {
				cat := t.Geti(c)
				//CHANGE from adaboost:
				if cat != m {
					t.Weights[c] = t.Weights[c] * math.Exp(weight) * t.Costs[cat]
				} else {
					t.Weights[c] = t.Weights[c] * math.Exp(-weight) * t.Costs[cat]
				}
			}

		}
	}
	normfactor := 0.0
	for _, v := range t.Weights {
		normfactor += v
	}
	for i, v := range t.Weights {
		t.Weights[i] = v / normfactor
	}
	return
}
