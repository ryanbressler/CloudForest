package CloudForest

import (
	"math/big"
)

const maxExhaustiveCats = 5
const maxNonRandomExahustive = 10
const maxNonBigCats = 30
const minImp = 1e-12

//FeatureI
type Feature interface {
	NCats() (n int)
	Length() (l int)
	IsMissing(i int) bool
	GoesLeft(i int, splitter *Splitter) bool
	PutMissing(i int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	FindPredicted(cases []int) (pred string)
	BestSplit(target Target,
		cases *[]int,
		parentImp float64,
		allocs *BestSplitAllocs) (bestNum float64, bestCat int, bestBigCat *big.Int, impurityDecrease float64)
	DecodeSplit(num float64, cat int, bigCat *big.Int) (s *Splitter)
	ShuffledCopy() (fake Feature)
	ImputeMissing()
	GetName() string
}

type NumFeature interface {
	Feature
	Get(i int) float64
	Put(i int, v float64)
	Mean(cases *[]int) float64
	Mode(cases *[]int) float64
	MeanSquaredError(cases *[]int, predicted float64) (e float64)
}

type CatFeature interface {
	Feature
	CatToNum(value string) (numericv int)
	NumToCat(i int) (value string)
	Geti(i int) int
	Puti(i int, v int)
	Modei(cases *[]int) int
	Get(i int) string
	Put(i int, v string)
	Mode(cases *[]int) string
	Gini(cases *[]int) float64
	GiniWithoutAlocate(cases *[]int, counts *[]int) (e float64)
}

//Target abstracts the methods needed for a feature to be predictable
//as either a catagroical or numerical feature in a random forest.
type Target interface {
	NCats() (n int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	FindPredicted(cases []int) (pred string)
}

//BoostingTarget
type BoostingTarget interface {
	NCats() (n int)
	SplitImpurity(l []int, r []int, counter *[]int) (impurityDecrease float64)
	Impurity(cases *[]int, counter *[]int) (impurity float64)
	Boost(partition *[][]int) (weight float64)
	FindPredicted(cases []int) (pred string)
}
