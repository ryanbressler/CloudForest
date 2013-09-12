package CloudForest

import (
	"math"
)

/*
WRFTarget wraps a numerical feature as a target for us in Adaptive Boosting (AdaBoost)
*/
type WRFTarget struct {
	CatFeature
	Weights []float64
}

/*
NewWRFTarget creates a categorical adaptive boosting target and initializes its weights.
*/
func NewWRFTarget(f CatFeature) (abt *WRFTarget, weights map[string]float64) {
	nCases := f.Length()
	abt = &WRFTarget{f, make([]float64, f.NCats())}

	for i, _ := range abt.Weights {
		abt.Weights[i] = weights[f.NumToCat(i)]
	}

	return
}

/*
WRFTarget.SplitImpurity is an AdaBoosting version of SplitImpurity.
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
		if !target.Missing[i] {
			cati := target.CatData[i]
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

//Find predicted takes the indexes of a set of cases and returns the
//predicted value. For categorical features this is a string containing the
//most common category and for numerical it is the mean of the values.
func (f *WRFTarget) FindPredicted(cases []int) (pred string) {

	counts := make([]int, f.NCats())
	for _, i := range *cases {
		if !f.Missing[i] {
			counts[f.CatData[i]] += 1
		}

	}
	max := 0
	for k, v := range counts {
		if v*f.Weights[k] > max {
			m = k
			max = v
		}
	}

	pred = f.Back[f.Modei(cases)]

	return

}
