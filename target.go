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
The ith entry in costs should contain the cost of misclassifying a case that actually
has the ith catagory.
*/
type RegretTarget struct {
	*Feature
	Costs []float64
}

//NewRegretTarget creates a RefretTarget and initializes RegretTarget.Costs to the proper length.
func NewRegretTarget(f *Feature) *RegretTarget {
	return &RegretTarget{f, make([]float64, f.NCats())}
}

/*RegretTarget.SetCosts puts costs in a map[string]float64 by feature name into the proper
entries in RegretTarget.Costs.*/
func (target *RegretTarget) SetCosts(costmap map[string]float64) {
	for i, c := range target.Back {
		target.Costs[i] = costmap[c]
	}
}

/*
RegretTarget.SplitImpurity is a version of Split Impurity that calls RegretTarget.Impurity
*/
func (target *RegretTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//RegretTarget.Impurity implements a simple regret functon that finds the average cost of
//a set using the misclasiffication costs in RegretTarget.Costs.
func (target *RegretTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Modei(cases)
	t := 0
	for _, c := range *cases {
		if target.Missing[c] == false {
			t += 1
			cat := target.CatData[c]
			if cat != m {
				e += target.Costs[cat]
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
	e = target.MeanL1Error(cases, m)
	return

}

//L1Target.MeanL1Error returns the  Mean L1 norm error of the cases specifed vs the predicted
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
