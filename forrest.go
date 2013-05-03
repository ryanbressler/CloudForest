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
it will be quicker to reimplment this method to write trees directelly to disk or grow
trees in parralel. See the growforest comand line utility for an example of this.

target is the feature to predict.

nSamples is the number of cases to sample (with replacment) for each tree.

mTry is the number of canidate features to evaluate at each node.

nTrees is the number of trees to grow.

leafSize is the minimum number of cases that should end up on a leaf.

itter indicates weather to use iterative spliting for all catagorical features or only those
with more then 6 catagories.

*/
func GrowRandomForest(fm *FeatureMatrix,
	target *Feature,
	nSamples int,
	mTry int,
	nTrees int,
	leafSize int,
	itter bool,
	splitmissing bool,
	importance *[]RunningMean) (f *Forest) {

	f = &Forest{target.Name, make([]*Tree, 0, nTrees)}

	//start with all features but the target as canidates
	canidates := make([]int, 0, len(fm.Data))
	targeti := fm.Map[target.Name]
	for i := 0; i < len(fm.Data); i++ {
		if i != targeti {
			canidates = append(canidates, i)
		}
	}

	//Slices for reuse during search for best spliter.
	l := make([]int, 0, nSamples)
	r := make([]int, 0, nSamples)
	var m *[]int
	if splitmissing {
		missing := make([]int, 0, nSamples)
		m = &missing
	}

	for i := 0; i < nTrees; i++ {
		nCases := len(fm.Data[0].Missing)
		cases := SampleWithReplacment(nSamples, nCases)

		f.Trees = append(f.Trees, NewTree())
		f.Trees[i].Grow(fm, target, cases, canidates, mTry, leafSize, itter, splitmissing, importance, &l, &r, m)
	}
	return
}
