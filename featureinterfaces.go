package CloudForest

import (
	"math/big"
)

//FeatureI
type FeatureI interface {
	NCats() (n int)
	Length() (l int)
	IsMissing(i int) bool
	PutMissing(i int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	FindPredicted(cases []int) (pred string)
	BestSplit(target Target,
		cases *[]int,
		parentImp float64,
		allocs *BestSplitAllocs) (bestNum float64, bestCat int, bestBigCat *big.Int, impurityDecrease float64)
	ShuffledCopy() (fake *Feature)
	ImputeMissing()
}

type NumFeature interface {
	FeatureI
	Get(i int) float64
	Put(i int, v float64)
	Mean(cases *[]int) float64
	Mode(cases *[]int) float64
	MeanSquaredError(cases *[]int, predicted float64) (e float64)
}

type CatFeature interface {
	FeatureI
	Geti(i int) int
	Puti(i int, v int)
	Modei(cases *[]int) int
	Get(i int) string
	Put(i int, v string)
	Mode(cases *[]int) string
	Gini(cases *[]int)
	GiniWithoutAlocate(cases *[]int, counts *[]int) (e float64)
}
