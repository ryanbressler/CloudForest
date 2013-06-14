package CloudForest

import (
	"fmt"
	"math"
)

/*
OrdinalTarget wraps a numerical feature as a target for us in ordinal regression.
Data should be represented as positive integers and the Error is embeded from the
embeded NumFeature.
*/
type OrdinalTarget struct {
	NumFeature
}

/*
OrdinalTarget.SplitImpurity is an ordinal version of SplitImpurity.
*/
func (target *OrdinalTarget) SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64) {
	nl := float64(len(l))
	nr := float64(len(r))

	impurityDecrease = nl * target.Impurity(&l, counter)
	impurityDecrease += nr * target.Impurity(&r, counter)

	impurityDecrease /= nl + nr
	return
}

func (f *OrdinalTarget) Predicted(cases *[]int) float64 {
	return f.Mode(cases)
}

func (f *OrdinalTarget) Mode(cases *[]int) (m float64) {
	counts := make(map[int]int)
	for _, i := range *cases {
		if !f.IsMissing(i) {
			counts[int(math.Floor(f.Get(i)+.5))] += 1
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
