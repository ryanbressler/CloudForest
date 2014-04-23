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

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *WRFTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	return target.SplitImpurity(l, r, m, allocs)
}

//Impurity is Gini impurity that uses the weights specified in WRFTarget.weights.
func (target *WRFTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	total := 0.0
	counts := *counter
	for i := range counts {
		counts[i] = 0
	}
	for _, i := range *cases {
		if !target.IsMissing(i) {
			cati := target.Geti(i)
			counts[cati]++
			total += target.Weights[cati]
		}
	}
	e = 1.0
	t := float64(total * total)
	for i, v := range counts {
		w := target.Weights[i]
		e -= float64(v*v) * w * w / t
	}
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
