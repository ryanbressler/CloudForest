package CloudForest

import (
	"fmt"
)

/*
OrdinalTarget wraps a numerical feature as a target for us in l1 norm regression.
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

//OrdinalTarget.Impurity is an ordinal version of impurity using Mode instead of Mean for prediction.
func (target *OrdinalTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	m := target.Predicted(cases)
	e = target.Error(cases, m)
	return

}

func (target *OrdinalTarget) FindPredicted(cases []int) (pred string) {
	fmt.Println("Ordinal")
	return fmt.Sprintf("%v", target.Predicted(&cases))
}
