package CloudForest

import ()

/*
GradBoostTarget wraps a numerical feature as a target for us in Gradiant Boosting Trees.

It should be used with the SumBallotBox.
*/
type GradBoostTarget struct {
	NumFeature
	LearnRate float64
	Mean      float64
}

func NewGradBoostTarget(f NumFeature, learnrate float64) (gbc *GradBoostTarget) {

	//res := NumFeature.(*DenseNumFeature).Copy().(*DenseNumFeature)
	sum := 0.0
	for i := 0; i < f.Length(); i++ {
		sum += f.Get(i)
	}

	// Set intial residual to
	prior := sum / float64(f.Length())

	for i := 0; i < f.Length(); i++ {
		v := f.Get(i) - prior
		f.Put(i, v)
	}

	//fmt.Println(res.Copy().(*DenseNumFeature).NumData)

	gbc = &GradBoostTarget{f, learnrate, prior}
	return

}

func (f *GradBoostTarget) Intercept() float64 {
	return f.Mean
}

//BUG(ryan) does GradBoostingTarget need seperate residuals and values?
func (f *GradBoostTarget) Boost(leaves *[][]int, preds *[]string) (weight float64) {
	for i, cases := range *leaves {
		f.Update(&cases, ParseFloat((*preds)[i]))
	}
	return f.LearnRate

}

//Update updates the underlying numeric data by subtracting the mean*weight of the
//specified cases from the value for those cases.
func (f *GradBoostTarget) Update(cases *[]int, predicted float64) {
	for _, i := range *cases {
		if !f.IsMissing(i) {
			f.Put(i, f.Get(i)-f.LearnRate*predicted)
		}
	}
}

//Impurity returns Gini impurity or mean squared error vs the mean for a set of cases
//depending on weather the feature is categorical or numerical
func (target *GradBoostTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	e = target.NumFeature.Impurity(cases, counter)
	if e <= minImp {
		return e
	}
	e = -1.0
	return e

}

func (target *GradBoostTarget) Sum(cases *[]int) (sum float64) {
	for _, i := range *cases {
		x := target.Get(i)
		sum += x
	}
	return
}

func FriedmanScore(allocs *BestSplitAllocs, l, r *[]int) (impurityDecrease float64) {
	nl := float64(len(*l))
	nr := float64(len(*r))
	diff := (allocs.Lsum / nl) - (allocs.Rsum / nr)
	impurityDecrease = (diff * diff * nl * nr) / (nl + nr)

	// if impurityDecrease <= 10e-6 {
	// 	impurityDecrease = 0.0
	// }
	return

}

// Friedman MSE slit improvment score from from equation 35 in "Greedy Function Approximation: A Gradiet Boosting Machine"
// Todo...what should the parent impurity be
func (target *GradBoostTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {

	allocs.Lsum = target.Sum(l)
	allocs.Rsum = target.Sum(r)

	impurityDecrease = FriedmanScore(allocs, l, r)
	return
}

func (target *GradBoostTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {

	MVsum := target.Sum(movedRtoL)

	allocs.Lsum += MVsum
	allocs.Rsum -= MVsum

	impurityDecrease = FriedmanScore(allocs, l, r)
	return
}
