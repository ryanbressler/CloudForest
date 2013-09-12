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

	for i, _ := range abt.Weights {
		abt.Weights[i] = weights[f.NumToCat(i)]
	}

	return
}

/*
WRFTarget.SplitImpurity is an weigtedRF version of SplitImpurity.
*/
func (target *WRFTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//WRFTarget.Impurity is Gini impurity that uses the weights specified in WRFTarget.weights.
func (target *WRFTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	total := 0.0
	counts := *counter
	for i, _ := range counts {
		counts[i] = 0
	}
	for _, i := range *cases {
		if !target.IsMissing(i) {
			cati := target.Geti(i)
			counts[cati] += 1
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
func (f *WRFTarget) FindPredicted(cases []int) (pred string) {

	counts := make([]int, f.NCats())
	for _, i := range cases {
		if !f.IsMissing(i) {
			counts[f.Geti(i)] += 1
		}

	}
	m := 0
	max := 0.0
	for k, v := range counts {
		val := float64(v) * f.Weights[k]
		if val > max {
			m = k
			max = val
		}
	}

	pred = f.NumToCat(m)

	return

}
