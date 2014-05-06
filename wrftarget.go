package CloudForest

/*
WRFTarget wraps a numerical feature as a target for us weigted random forest.
*/
type WRFTarget struct {
	CatFeature
	Weights []float64
}

/*
NewWRFTarget creates a weighted random forest target and initializes its weights.
*/
func NewWRFTarget(f CatFeature, weights map[string]float64) (abt *WRFTarget) {
	abt = &WRFTarget{f, make([]float64, f.NCats())}

	for i := range abt.Weights {
		abt.Weights[i] = weights[f.NumToCat(i)]
	}

	return
}

/*
SplitImpurity is an weigtedRF version of SplitImpurity.
*/
func (target *WRFTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
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
func (target *WRFTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
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

	impurityDecrease = nl * target.ImpFromCounts(allocs.LCounter)
	impurityDecrease += nr * target.ImpFromCounts(allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.ImpFromCounts(allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//Impurity is Gini impurity that uses the weights specified in WRFTarget.weights.
func (target *WRFTarget) Impurity(cases *[]int, counter *[]int) (e float64) {

	target.CountPerCat(cases, counter)

	return target.ImpFromCounts(counter)
}

//ImpFromCounts recalculates gini impurity from class counts for us in intertive updates.
func (target *WRFTarget) ImpFromCounts(counter *[]int) (e float64) {

	total := 0.0
	for i, v := range *counter {
		w := target.Weights[i]
		total += float64(v) * w

		e -= float64(v*v) * w * w
	}

	e /= float64(total * total)
	e++

	return

}

//FindPredicted finds the predicted target as the weighted catagorical Mode.
func (target *WRFTarget) FindPredicted(cases []int) (pred string) {

	counts := make([]int, target.NCats())
	for _, i := range cases {
		if !target.IsMissing(i) {
			counts[target.Geti(i)] += 1
		}

	}
	m := 0
	max := 0.0
	for k, v := range counts {
		val := float64(v) * target.Weights[k]
		if val > max {
			m = k
			max = val
		}
	}

	pred = target.NumToCat(m)

	return

}
