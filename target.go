package CloudForest

import ()

//Target abstracts the methods needed for a feature to be predictable.
type Target interface {
	NCats() (n int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	FindPredicted(cases []int) (pred string)
}
