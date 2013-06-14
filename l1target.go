package CloudForest

import (
	"math"
)

/*
L1Target wraps a numerical feature as a target for us in l1 norm regression.
*/
type L1Target struct {
	NumFeature
}

/*
L1Target.SplitImpurity is an L1 version of SplitImpurity.
*/
func (target *L1Target) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//L1Target.Impurity is an L1 version of impurity returning L1 instead of squared error.
func (target *L1Target) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Mean(cases)
	e = target.Error(cases, m)
	return

}

func (f *L1Target) Norm(i int, v float64) float64 {
	return math.Abs(f.Get(i) - v)
}

//L1Target.MeanL1Error returns the  Mean L1 norm error of the cases specified vs the predicted
//value. Only non missing cases are considered.
func (target *L1Target) Error(cases *[]int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range *cases {
		if !target.IsMissing(i) {
			e += math.Abs(predicted - target.Get(i))
			n += 1
		}

	}
	e = e / float64(n)
	return

}
