package CloudForest

import (
	"math"
)

//Target abstracts the methods needed for a feature to be predictable.
type Target interface {
	NCats() (n int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	FindPredicted(cases []int) (pred string)
}

/*
RegretTarget wraps a catagorical feature for use in regret driven classification.
*/
type RegretTarget struct {
	*Feature
	costs []float64
}

/*
SplitImpurity calculates the impurity of a splitinto the specified left and right
groups. This is depined as pLi*(tL)+pR*i(tR) where pL and pR are the probability of case going left or right
and i(tl) i(tR) are the left and right impurites.

Counter is only used for catagorical targets and should have the same length as the number of catagories in the target.
*/
func (target *RegretTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

func (target *RegretTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Modei(cases)
	t := 0
	for _, c := range *cases {
		if target.Missing[c] == false {
			t += 1
			cat := target.CatData[c]
			if cat != m {
				e += target.costs[cat]
			}
		}

	}
	e /= float64(t)

	return
}

/*
L1Target wraps a numerical feature as a target for us in l1 norm regresion.
*/
type L1Target struct {
	*Feature
}

/*
SplitImpurity calculates the impurity of a splitinto the specified left and right
groups. This is depined as pLi*(tL)+pR*i(tR) where pL and pR are the probability of case going left or right
and i(tl) i(tR) are the left and right impurites.

Counter is only used for catagorical targets and should have the same length as the number of catagories in the target.
*/
func (target *L1Target) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//Impurity returns Gini impurity or mean squared error vs the mean for a set of cases
//depending on weather the feature is catagorical or numerical
func (target *L1Target) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Mean(cases)
	e = target.MeanL1Error(cases, m)
	return

}

//MeanL1Error returns the  Mean L1 norm error of the cases specifed vs the predicted
//value. Only non missing casses are considered.
func (target *L1Target) MeanL1Error(cases *[]int, predicted float64) (e float64) {
	e = 0.0
	n := 0
	for _, i := range *cases {
		if !target.Missing[i] {
			e += math.Abs(predicted - target.NumData[i])
			n += 1
		}

	}
	e = e / float64(n)
	return

}
