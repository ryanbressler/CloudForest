package CloudForest

import (
	"fmt"
)

/*
HDistanceTarget wraps a categorical feature for use in Hellinger Distance tree
growth.
*/
type HDistanceTarget struct {
	CatFeature
	Pos_class string
}

//NewHDistanceTarget creates a RefretTarget and initializes HDistanceTarget.Costs to the proper length.
func NewHDistanceTarget(f CatFeature, pos_class string) *HDistanceTarget {
	return &HDistanceTarget{f, pos_class}
}

/*
HDistanceTarget.SplitImpurity is a version of Split Impurity that calls HDistanceTarget.Impurity
*/
func (target *HDistanceTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) float64 {
	target.CountPerCat(l, allocs.LCounter)
	target.CountPerCat(r, allocs.RCounter)

	return target.HDist(allocs.LCounter, allocs.RCounter)
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *HDistanceTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) float64 {
	target.MoveCountsRtoL(allocs, movedRtoL)
	return target.HDist(allocs.LCounter, allocs.RCounter)
}

func (target *HDistanceTarget) HDist(lcounts *[]int, rcounts *[]int) (d float64) {
	l := *lcounts
	r := *rcounts

	// Hellinger Distance = sqrt
	// (count(1, left)/count(1) - count(0, left)/count0)^2
	// (count(1, right)/count(1) - count(0, right)/count0)^2

	total_0 := float64(l[0] + r[0])
	total_1 := float64(l[1] + r[1])

	inner := float64(l[0])
	inner /= total_0
	inner -= float64(l[1]) / total_1
	d = inner * inner

	inner = float64(r[0])
	inner /= total_0
	inner -= float64(r[1]) / total_1
	d += inner * inner

	// not needed because monotonic
	// d = math.Sqrt(d)
	return

}

func (target *HDistanceTarget) FindPredicted(cases []int) (pred string) {
	// TODO(ryan): lapalcian smoothing?
	prob_true := 0.0
	t := target.CatToNum(target.Pos_class)

	count := 0
	for _, i := range cases {
		if target.Geti(i) == t {
			count++
		}

	}
	prob_true = float64(count) / float64(len(cases))

	return fmt.Sprintf("%v", prob_true)
}

//HDistanceTarget.Impurity
func (target *HDistanceTarget) Impurity(cases *[]int, counts *[]int) (e float64) {

	return -1.0

}
