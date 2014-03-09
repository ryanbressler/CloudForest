package CloudForest

import ()

//Forest represents a collection of decision trees grown to predict Target.
type Forest struct {
	//Forest string
	Target string
	Trees  []*Tree
}

/*
GrowRandomForest grows a forest using Brieman and Cutler's method. For many cases it
it will yield better performance to re-implment this method to write trees directly to disk or grow
trees in parallel. See the grow forest command line utility for an example of this.

target is the feature to predict.

nSamples is the number of cases to sample (with replacement) for each tree.

mTry is the number of candidate features to evaluate at each node.

nTrees is the number of trees to grow.

leafSize is the minimum number of cases that should end up on a leaf.

itter indicates weather to use iterative splitting for all categorical features or only those
with more then 6 categories.
*/

func GrowRandomForest(fm *FeatureMatrix,
	target Feature,
	nSamples int,
	mTry int,
	nTrees int,
	leafSize int,
	splitmissing bool,
	vet bool,
	importance *[]*RunningMean) (f *Forest) {

	f = &Forest{target.GetName(), make([]*Tree, 0, nTrees)}

	//start with all features but the target as candidates
	candidates := make([]int, 0, len(fm.Data))
	targeti := fm.Map[f.Target]
	for i := 0; i < len(fm.Data); i++ {
		if i != targeti {
			candidates = append(candidates, i)
		}
	}

	//Slices for reuse during search for best splitter.
	allocs := NewBestSplitAllocs(nSamples, target)

	for i := 0; i < nTrees; i++ {
		nCases := fm.Data[0].Length()
		cases := SampleWithReplacment(nSamples, nCases)

		f.Trees = append(f.Trees, NewTree())
		f.Trees[i].Grow(fm, target, cases, candidates, nil, mTry, leafSize, splitmissing, vet, false, importance, nil, allocs)
		switch target.(type) {
		case BoostingTarget:
			f.Trees[i].Weight = target.(BoostingTarget).Boost(f.Trees[i].Partition(fm))
		}
	}
	return
}
