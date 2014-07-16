package CloudForest

import ()

/*
TransTarget is used for density estimating trees. It contains a set of features and the
count of cases.
*/
type TransTarget struct {
	CatFeature
	Features  *[]Feature
	Unlabeled int
	Alpha     float64
	N         int
	MaxCats   int
}

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

//TransTarget.Impurity uses the impurity measure defined in "Density Estimating Trees"
//by Parikshit Ram and Alexander G. Gray
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

//TransTarget.FindPredicted returns the string representation of the density in the region
//spaned by the specified cases.
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
