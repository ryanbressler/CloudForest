package CloudForest

import ()

/*
TransTarget is used for semi supervised transduction trees [1] that balance compine supervised impurity with
a purelly density based term.

I = I_supervised + alpha * I_unsupervised

I_supervised is called from the embeded CatFeature so that it can be Gini, Entropy, Weighted or any other
of the existing non-boosting impurities. Boosting impurities could be implemented with minimal work.

I_unsupervised uses a density estimating term that differs from the one described in [1] and is instead
based on the technique described in [2] which avoids some assumptions and allows a simple implementation.

[1] A. Criminisi, J. Shotton, and E. Konukoglu, "Decision Forests for Classification, Regression,
Density Estimation, Manifold Learning and Semi-Supervised Learning"
Microsoft Research technical report TR-2011-114

[2] Parikshit Ram, Alexander G. Gray, Density Estimation Trees
http://research.microsoft.com/pubs/155552/decisionForests_MSR_TR_2011_114.pdf

One diffrence from [1] is that the unlabelled class is considered a standard class for I_supervised
to allow once class problems.
*/
type TransTarget struct {
	CatFeature
	Features  *[]Feature
	Unlabeled int
	Alpha     float64
	N         int
	MaxCats   int
}

/*NewTransTarget returns a TransTarget using the supervised Impurity from the provided CatFeature t,
Density in the specified Features fm (excluding any with the same name as t), considering the class label
provided in "unlabeled" as unlabeled for transduction. Alpha is the weight of the unspervised term relative to
the supervised and ncases is the number of cases that will be called at the root of the tree (may be depreciated as not needed).
*/
func NewTransTarget(t CatFeature, fm *[]Feature, unlabeled string, alpha float64, ncases int) *TransTarget {
	maxcats := 0
	for _, f := range *fm {
		if f.NCats() > maxcats {
			maxcats = f.NCats()
		}
	}

	return &TransTarget{t, fm, t.CatToNum(unlabeled), alpha, ncases, maxcats}

}

/*
TransTarget.SplitImpurity is a density estimating version of SplitImpurity.
*/
func (target *TransTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.Impurity(l, allocs.LCounter)
	impurityDecrease += nr * target.Impurity(r, allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.Impurity(m, allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *TransTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	return target.SplitImpurity(l, r, m, allocs)
}

func (target *TransTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	//TODO: filter out unlabeled cases from the call to target.CatFeature.Impurity at least for
	//multiclass problems
	return target.CatFeature.Impurity(cases, counter) + target.Alpha*target.Density(cases, counter)
}

/*TransTarget.Density uses an impurity designed to maximize the density within each side of the split
based on the method in "Density Estimating Trees" by Parikshit Ram and Alexander G. Gray.
It loops over all of the non target features and for the ones with non zero span calculates product(span_i)/(t*t)
where t is the number of cases.

Refinements to this method might include t*t->t^n where n is the number of features with
non zero span or other changes to how zero span features are handeled. I also suspect that this method
handles numerical features for which diffrent splits will have diffrent total spans based on the
distance between the points on either side of the split point better then categorical
features for which the total span of a split will allways be the number of categories.

The origional paper also included N which is not used here.*/
func (target *TransTarget) Density(cases *[]int, counter *[]int) (e float64) {
	t := len(*cases)
	//e = float64(t*t) / float64(target.N*target.N)
	e = 1 / float64(t*t) // float64(target.N*target.N)
	span := 0.0
	bigenoughcounter := make([]int, target.MaxCats, target.MaxCats)
	targetname := target.GetName()

	for _, f := range *target.Features {
		if f.GetName() != targetname {

			span = f.Span(cases, &bigenoughcounter)

			if span > 0.0 {
				e *= span
			}

			ncats := f.NCats()
			for i := 0; i < ncats; i++ {
				bigenoughcounter[i] = 0
			}

		}
	}

	return
}

//TransTarget.FindPredicted returns the prediction of the specified cases which is the majority
//class that is not the unlabeled class. A set of cases will only be predicted to be the ulabeled
//class if has no labeled points.
func (target *TransTarget) FindPredicted(cases []int) string {
	counts := make([]int, target.NCats())
	for _, i := range cases {

		counts[target.Geti(i)] += 1

	}
	max := 0
	m := target.Unlabeled
	for k, v := range counts {
		if v > max && k != target.Unlabeled {
			m = k
			max = v
		}
	}

	// if counts[target.Unlabeled] > 10*max {
	// 	m = target.Unlabeled
	// }

	return target.NumToCat(m)
}
