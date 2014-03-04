package CloudForest

import (
	"fmt"
)

/*
OrdinalTarget wraps a numerical feature as a target for us in ordinal regression.
Data should be represented as positive integers and the Error is embeded from the
embeded NumFeature.
*/
type OrdinalTarget struct {
	NumFeature
	nClass int
	max    float64
}

/*
NewOrdinalTarget creates a categorical adaptive boosting target and initializes its weights.
*/
func NewOrdinalTarget(f NumFeature) (abt *OrdinalTarget) {
	nCases := f.Length()
	abt = &OrdinalTarget{f, 0, 0.0}
	for i := 0; i < nCases; i++ {
		v := f.Get(i)
		if v > abt.max {
			abt.max = v
		}
	}

	abt.nClass = int(abt.max) + 1
	return
}

/*
OrdinalTarget.SplitImpurity is an ordinal version of SplitImpurity.
*/
func (target *OrdinalTarget) SplitImpurity(l []int, r []int, m []int, allocs *BestSplitAllocs) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))
	nm := 0.0

	impurityDecrease = nl * target.Impurity(&l, nil)
	impurityDecrease += nr * target.Impurity(&r, nil)
	if m != nil {
		nm := float64(len(m))
		impurityDecrease += nm * target.Impurity(&m, nil)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *OrdinalTarget) UpdateSImpFromAllocs(l []int, r []int, m []int, allocs *BestSplitAllocs, movedRtoL []int) (impurityDecrease float64) {
	return target.SplitImpurity(l, r, m, allocs)
}

func (f *OrdinalTarget) Predicted(cases *[]int) float64 {
	return f.Mode(cases)
}

func (f *OrdinalTarget) Mode(cases *[]int) (m float64) {
	counts := make([]int, f.nClass)
	for _, i := range *cases {
		if !f.IsMissing(i) {
			counts[int(f.Get(i))] += 1
		}

	}
	max := 0
	for k, v := range counts {
		if v > max {
			m = float64(k)
			max = v
		}
	}
	return

}

//OrdinalTarget.Impurity is an ordinal version of impurity using Mode instead of Mean for prediction.
func (target *OrdinalTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Predicted(cases)
	e = target.Error(cases, m)
	return

}

func (target *OrdinalTarget) FindPredicted(cases []int) (pred string) {
	return fmt.Sprintf("%v", target.Predicted(&cases))
}
