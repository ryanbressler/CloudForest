package CloudForest

import (
	"math"
)

/*
NumNumAdaBoostTarget wraps a numerical feature as a target for us in Adaptive Boosting (AdaBoost)
*/
type NumAdaBoostTarget struct {
	NumFeature
	Weights []float64
}

func NewNumAdaBoostTarget(f NumFeature) (abt *NumAdaBoostTarget) {
	nCases := f.Length()
	abt = &NumAdaBoostTarget{f, make([]float64, nCases)}
	for i, _ := range abt.Weights {
		abt.Weights[i] = 1 / float64(nCases)
	}
	return
}

/*
NumAdaBoostTarget.SplitImpurity is an AdaBoosting version of SplitImpurity.
*/
func (target *NumAdaBoostTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//NumAdaBoostTarget.Impurity is an AdaBoosting that uses the weights specified in NumAdaBoostTarget.weights.
func (target *NumAdaBoostTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	e = 0.0
	m := target.Predicted(cases)
	for _, c := range *cases {
		if target.IsMissing(c) == false {

			e += target.Weights[c] * target.Error(&[]int{c}, m)

		}

	}
	return
}

func (t *NumAdaBoostTarget) Boost(leaves *[][]int) (weight float64) {
	weight = 0.0
	for _, cases := range *leaves {
		weight += t.Impurity(&cases, nil)
	}
	if weight >= .5 {
		return 0.0
	}
	weight = .5 * math.Log((1-weight)/weight)

	for _, cases := range *leaves {
		m := t.Predicted(&cases)
		for _, c := range cases {
			if t.IsMissing(c) == false {
				t.Weights[c] = t.Weights[c] * math.Exp(t.Error(&[]int{c}, m)*weight)
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
