package CloudForest

import (
	"fmt"
	"math"
)

func Logit(x float64) float64 {
	return math.Log(x / (1.0 - x))
}

func Expit(x float64) (out float64) {
	//return 1.0 / (1.0 + math.Exp(-1.0*x))
	out = 0.5 * x
	out = math.Tanh(out)
	out += 1.0
	out *= 0.5
	return out
}

/*
GradBoostClassTarget wraps a numerical feature as a target for us in Two Class Gradiant Boosting Trees.

It should be used with SumBallotBox and expit transformed to get class probabilities.
*/
type GradBoostClassTarget struct {
	*GradBoostTarget
	Actual    NumFeature
	Pred      NumFeature
	LearnRate float64
	Prior     float64
	Pos_class string
}

func NewGradBoostClassTarget(f CatFeature, learnrate float64, pos_class string) (gbc *GradBoostClassTarget) {

	//fmt.Println("Back: ", f.CatToNum(pos_class), f.(*DenseCatFeature).Back)

	actual := f.EncodeToNum()[0].(*DenseNumFeature)
	pred := actual.Copy().(*DenseNumFeature)
	// Make sure the encoding has the positive class as 1
	for i := 0; i < f.Length(); i++ {
		if f.GetStr(i) == pos_class {
			actual.Put(i, 1.0)
		} else {
			actual.Put(i, 0.0)
		}

	}

	res := &GradBoostTarget{actual.Copy().(*DenseNumFeature), learnrate, 0.0}

	pos := 0.0
	for i := 0; i < actual.Length(); i++ {
		pos += actual.Get(i)
	}

	// Set intial residual to
	prior := math.Log(pos / (float64(res.Length()) - pos))

	for i := 0; i < res.Length(); i++ {
		pred.Put(i, prior)
		v := actual.Get(i) - Expit(prior)
		res.Put(i, v)
	}

	//fmt.Println(res.Copy().(*DenseNumFeature).NumData)

	gbc = &GradBoostClassTarget{res, actual, pred, learnrate, prior, pos_class}
	return

}

func (f *GradBoostClassTarget) Intercept() float64 {
	return f.Prior
}

//BUG(ryan) does GradBoostingTarget need seperate residuals and values?
func (f *GradBoostClassTarget) Boost(leaves *[][]int, preds *[]string) (weight float64) {
	for i, cases := range *leaves {
		f.Update(&cases, ParseFloat((*preds)[i]))
	}
	return f.LearnRate

}

func (f *GradBoostClassTarget) Predicted(cases *[]int) float64 {
	//TODO(ryan): update predicted on whole data not just in bag
	num := 0.0
	denom := 0.0

	for _, c := range *cases {
		r := f.Get(c)
		num += r
		y := f.Actual.Get(c)
		denom += (y - r) * (1.0 - y + r)

	}

	return num / denom // 1.0 / (1.0 + math.Exp(-1*meanlogodds))
}

func (f *GradBoostClassTarget) FindPredicted(cases []int) (pred string) {
	pred = fmt.Sprintf("%v", f.Predicted(&cases))
	return

}

//Update updates the underlying numeric data by subtracting the mean*weight of the
//specified cases from the value for those cases.
func (f *GradBoostClassTarget) Update(cases *[]int, predicted float64) {
	for _, i := range *cases {
		pred := f.Pred.Get(i) + f.LearnRate*predicted
		f.Pred.Put(i, pred)

		g := f.Actual.Get(i) - Expit(pred)
		f.Put(i, g)

	}
}
