package CloudForest

import ()

//BoostingTarget
type BoostingTarget interface {
	NCats() (n int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	Boost(partition *[][]int) (weight float64)
	FindPredicted(cases []int) (pred string)
}
