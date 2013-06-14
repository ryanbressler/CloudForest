package CloudForest

import (
	"math"
)

/*
NumNumAdaBoostTarget wraps a numerical feature as a target for us in (Experimental) Adaptive Boosting
Regression.
*/
type NumAdaBoostTarget struct {
	NumFeature
	Weights    []float64
	NormFactor float64
}

func NewNumAdaBoostTarget(f NumFeature) (abt *NumAdaBoostTarget) {
	nCases := f.Length()
	abt = &NumAdaBoostTarget{f, make([]float64, nCases), 0.0}
	cases := make([]int, nCases)
	for i, _ := range abt.Weights {
		abt.Weights[i] = 1 / float64(nCases)
		cases[i] = i
	}
	abt.NormFactor = abt.Impurity(&cases, nil) * float64(nCases)
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
	singlecase := (*cases)[0:0]
	for i, c := range *cases {
		if target.IsMissing(c) == false {
			singlecase = (*cases)[i : i+1]
			e += target.Weights[c] * target.Error(&singlecase, m)

		}

	}
	return
}

//AdaBoostTarget.Boost performs numerical adaptive boosting using the specified partition and
//returns the weight that tree that generated the partition should be given.
//Trees with error greater then the impurity of the total feature (NormFactor) times the number
//of partions are given zero weight. Other trees have tree weight set to:
//
// weight = math.Log(1 / norm)
//
//and weights updated to:
//
// t.Weights[c] = t.Weights[c] * math.Exp(t.Error(&[]int{c}, m)*weight)
//
//These functions are chosen to provide a rough analog to catagorical adaptive boosting for
//numerical data with unbounded error.
func (t *NumAdaBoostTarget) Boost(leaves *[][]int) (weight float64) {
	if len(*leaves) == 0 {
		return 0.0
	}
	imp := 0.0
	//nCases := 0
	for _, cases := range *leaves {
		imp += t.Impurity(&cases, nil)
		//nCases += len(cases)
	}
	norm := t.NormFactor
	if imp > norm {
		return 0.0
	}

	weight = math.Log(norm / imp)

	for _, cases := range *leaves {
		m := t.Predicted(&cases)
		singlecase := (cases)[0:0]
		for i, c := range cases {
			if t.IsMissing(c) == false {
				singlecase = (cases)[i : i+1]
				t.Weights[c] = t.Weights[c] * math.Exp(weight*(t.Error(&singlecase, m)-imp))
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
