package CloudForest

import (
	"math"
	"strconv"

	"github.com/drewlanenga/govector"
)

var (
	mTryRegression     = func(x int) int { return int(math.Ceil(float64(x) / 3.0)) }
	mTryClassification = func(x int) int { return int(math.Ceil(math.Sqrt(float64(x)))) }
)

//Forest represents a collection of decision trees grown to predict Target.
type Forest struct {
	//Forest string
	Target    string
	Trees     []*Tree
	Intercept float64
}

type FullForest struct {
	Forest
	InBag       [][]float64
	Predictions []float64
	Importance  *[]*RunningMean
}

type ForestConfig struct {
	NSamples     int
	NTrees       int
	MTry         int
	LeafSize     int
	MaxDepth     int
	SplitMissing bool
	Force        bool
	Vet          bool
	EvalOOB      bool
	Replace      bool
	InBag        bool
}

func NewClassificationConfig(n int, replace bool) *ForestConfig {
	return newForestConfig(n, mTryClassification(n), replace)
}

func NewRegressionConfig(n int, replace bool) *ForestConfig {
	return newForestConfig(n, mTryRegression(n), replace)
}

func newForestConfig(n, mtry int, replace bool) *ForestConfig {
	nsamples := n
	if !replace {
		nsamples = int(math.Ceil(0.632 * float64(n)))
	}

	return &ForestConfig{
		NSamples: nsamples,
		NTrees:   500,
		LeafSize: 1,
		MTry:     mtry,
	}
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

func GrowRandomForest(fm *FeatureMatrix, target Target, config *ForestConfig) *FullForest {
	var (
		targetName = target.GetName()
		trees      = make([]*Tree, config.NTrees)
		inbag      = make([][]float64, config.NTrees)
		candidates = make([]int, 0, len(fm.Data)-1)
		allocs     = NewBestSplitAllocs(config.NSamples, target)
		importance = NewRunningMeans(config.NSamples)
	)

	for i, feature := range fm.Data {
		if feature.GetName() != targetName {
			candidates = append(candidates, i)
		}
	}

	for i := 0; i < config.NTrees; i++ {
		nCases := fm.Data[0].Length()

		var cases []int
		if config.Replace {
			cases = SampleWithReplacment(config.NSamples, nCases)
		} else {
			cases = SampleWithoutReplacement(config.NSamples, nCases)
		}

		if config.InBag {
			inbag[i] = make([]float64, config.NSamples)
			for _, c := range cases {
				inbag[i][c]++
			}
		}

		tree := NewTree()
		tree.Target = targetName
		trees[i] = tree
		trees[i].Grow(
			fm,
			target,
			cases,
			candidates,
			nil,
			config.MTry,
			config.LeafSize,
			config.MaxDepth,
			config.SplitMissing,
			config.Force,
			config.Vet,
			config.EvalOOB,
			false,
			importance,
			nil,
			allocs,
		)
	}

	f := &FullForest{
		Forest: Forest{
			Trees:  trees,
			Target: targetName,
		},
	}

	if config.InBag {
		f.InBag = inbag
		f.Predictions = f.PredictAll(fm)
	}

	return f
}

func (f *Forest) PredictAll(fm *FeatureMatrix) []float64 {
	cases := makeCases(fm.Data[0].Length())
	predictions := make([]float64, len(f.Trees))

	for i, tree := range f.Trees {
		tree.Root.Recurse(func(n *Node, cases []int, depth int) {
			if n.Left == nil && n.Right == nil {
				if val, err := strconv.ParseFloat(n.Pred, 64); err == nil {
					predictions[i] = val
				} else {
					predictions[i] = math.NaN()
				}
			}
		}, fm, cases, 0)
	}

	return predictions
}

func JackKnife(predictions []float64, inbag [][]float64) float64 {
	var (
		n       = len(inbag)
		B       = float64(len(inbag))
		B2      = B * B
		sum     = float64(0)
		preds   = govector.Vector(predictions)
		avgPred = preds.Mean()
	)

	preds = preds.Apply(func(f float64) float64 { return f - avgPred })

	for i := 0; i < n; i++ {
		val, err := govector.DotProduct(govector.Vector(inbag[i]), preds)
		if err != nil {
			continue
		}

		val *= val
		val /= B2
		sum += val
	}
	return sum

	/*

	   #
	   # Apply Monte Carlo bias correction
	   #

	   N.var = mean(Matrix::rowMeans(N^2) - Matrix::rowMeans(N)^2)
	   boot.var = rowSums(pred.centered^2) / B
	   bias.correction = n * N.var * boot.var / B
	   vars = raw.IJ - bias.correction
	*/
}
