package CloudForest

/*
RegretTarget wraps a categorical feature for use in regret driven classification.
The ith entry in costs should contain the cost of misclassifying a case that actually
has the ith category.

It is roughly equivelent to the ideas presented in:

http://machinelearning.wustl.edu/mlpapers/paper_files/icml2004_LingYWZ04.pdf

"Decision Trees with Minimal Costs"
Charles X. Ling,Qiang Yang,Jianning Wang,Shichao Zhang
*/
type RegretTarget struct {
	CatFeature
	Costs []float64
}

//NewRegretTarget creates a RefretTarget and initializes RegretTarget.Costs to the proper length.
func NewRegretTarget(f CatFeature) *RegretTarget {
	return &RegretTarget{f, make([]float64, f.NCats())}
}

/*RegretTarget.SetCosts puts costs in a map[string]float64 by feature name into the proper
entries in RegretTarget.Costs.*/
func (target *RegretTarget) SetCosts(costmap map[string]float64) {
	for i := 0; i < target.NCats(); i++ {
		c := target.NumToCat(i)
		target.Costs[i] = costmap[c]
	}
}

/*
RegretTarget.SplitImpurity is a version of Split Impurity that calls RegretTarget.Impurity
*/
func (target *RegretTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
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
func (target *RegretTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
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
func (target *RegretTarget) FindPredicted(cases []int) (pred string) {

	mi := 0
	mc := 0.0
	counts := make([]int, target.NCats())

	target.CountPerCat(&cases, &counts)

	for cat, count := range counts {
		cc := float64(count) * target.Costs[cat]
		if cc > mc {
			mi = cat
			mc = cc
		}
	}

	return target.NumToCat(mi)

}

//ImpFromCounts recalculates gini impurity from class counts for us in intertive updates.
func (target *RegretTarget) ImpFromCounts(t int, counter *[]int) (e float64) {

	mi := 0

	mc := 0.0

	for cat, count := range *counter {
		cc := float64(count) * target.Costs[cat]

		if cc > mc {
			mi = cat
			mc = cc
		}

	}

	for cat, count := range *counter {

		t += count
		if cat != mi {
			e += target.Costs[cat] * float64(count)
		}

	}
	e /= float64(t)

	return

}

//Impurity implements an impurity based on misslassification costs.
func (target *RegretTarget) Impurity(cases *[]int, counter *[]int) (e float64) {

	target.CountPerCat(cases, counter)
	t := len(*cases)
	e = target.ImpFromCounts(t, counter)

	return

}

//RegretTarget.Impurity implements a simple regret function that finds the average cost of
//a set using the misclassification costs in RegretTarget.Costs.
// func (target *RegretTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
// 	m := target.Modei(cases)
// 	t := 0
// 	for _, c := range *cases {
// 		if target.IsMissing(c) == false {
// 			t += 1
// 			cat := target.Geti(c)
// 			if cat != m {
// 				e += target.Costs[cat]
// 			}
// 		}

// 	}
// 	e /= float64(t)

// 	return
// }
