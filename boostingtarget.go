package CloudForest

import ()

//BoostingTarget
type BoostingTarget interface {
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	UpdateResiduals(cases *[]int)
	FindPredicted(cases []int) (pred string)
}
