package CloudForest

import (
	"math"
)

/*
AdaBoostTarget wraps a numerical feature as a target for us in Adaptive Boosting (AdaBoost)
*/
type AdaBoostTarget struct {
	*Feature
	Weights []float64
}

func NewAdaBoostTarget(f *Feature) (abt *AdaBoostTarget) {
	nCases := len(f.CatData)
	abt = &AdaBoostTarget{f, make([]float64, nCases)}
	for i, _ := range abt.Weights {
		abt.Weights[i] = 1 / float64(nCases)
	}
	return
}

/*
AdaBoostTarget.SplitImpurity is an AdaBoosting version of SplitImpurity.
*/
func (target *AdaBoostTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

//AdaBoostTarget.Impurity is an AdaBoosting that uses the weights specified in AdaBoostTarget.weights.
func (target *AdaBoostTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	e = 0.0
	m := target.Modei(cases)
	for _, c := range *cases {
		if target.Missing[c] == false {
			cat := target.CatData[c]
			if cat != m {
				e += target.Weights[c]
			}
		}

	}
	return
}

func (t *AdaBoostTarget) Boost(leaves *[][]int) (weight float64) {
	weight = 0.0
	for _, cases := range *leaves {
		weight += t.Impurity(&cases, nil)
	}
	if weight >= .5 {
		return 0.0
	}
	weight = .5 * math.Log((1-weight)/weight)

	for _, cases := range *leaves {
		m := t.Modei(&cases)
		for _, c := range cases {
			if t.Missing[c] == false {
				cat := t.CatData[c]
				if cat != m {
					t.Weights[c] = t.Weights[c] * math.Exp(weight)
				} else {
					t.Weights[c] = t.Weights[c] * math.Exp(-weight)
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
