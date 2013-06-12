package CloudForest

import ()

/*
GradBoostTarget wraps a numerical feature as a target for us in Adaptive Boosting (AdaBoost)
*/
type GradBoostTarget struct {
	*Feature
	LearnRate float64
}

//BUG(ryan) does GradBoostingTarget need seperate residuals and values?
func (f *GradBoostTarget) Boost(leaves *[][]int) (weight float64) {
	for _, cases := range *leaves {
		f.Update(&cases)
	}
	return f.LearnRate

}

//Update updates the underlying numeric data by subtracting the mean*weight of the
//specified cases from the value for those cases.
func (f *GradBoostTarget) Update(cases *[]int) {
	m := f.Mean(cases)
	for v, _ := range *cases {
		f.NumData[v] -= f.LearnRate * m
	}
}
