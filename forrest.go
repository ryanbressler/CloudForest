package CloudForest

import (
	"fmt"
	"math"
	"strconv"
	"sync"

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

type ForestModel struct {
	Forest      *Forest
	InBag       [][]float64
	Predictions [][]float64
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

func GrowRandomForest(fm *FeatureMatrix, target Target, config *ForestConfig) *ForestModel {
	var (
		targetName = target.GetName()
		allocs     = NewBestSplitAllocs(config.NSamples, target)
		importance = NewRunningMeans(config.NSamples)
		nCases     = fm.Data[0].Length()
		trees      = make([]*Tree, config.NTrees)
		inbag      = make([][]float64, config.NTrees)
		candidates = make([]int, 0, len(fm.Data)-1)
	)

	// construct list of candidate features
	for i, feature := range fm.Data {
		if feature.GetName() != targetName {
			candidates = append(candidates, i)
		}
	}

	for i := 0; i < config.NTrees; i++ {
		var cases []int
		if config.Replace {
			cases = SampleWithReplacment(config.NSamples, nCases)
		} else {
			cases = SampleWithoutReplacement(config.NSamples, nCases)
		}

		// if in-bag, keep track of the cases
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

	forest := &Forest{Trees: trees, Target: targetName}
	full := &ForestModel{Forest: forest, Importance: importance}

	if config.InBag {
		full.InBag = inbag
		full.Predictions = forest.PredictAll(fm)
	}

	return full
}

// PredictAll returns the predictions
func (f *Forest) PredictAll(fm *FeatureMatrix) [][]float64 {
	n := fm.Data[0].Length()
	cases := makeCases(n)
	predictions := make([][]float64, n)

	for i := range predictions {
		predictions[i] = make([]float64, len(f.Trees))
	}

	for i, tree := range f.Trees {
		tree.Root.Recurse(func(n *Node, cases []int, depth int) {
			if n.Left == nil && n.Right == nil {
				for _, j := range cases {
					if val, err := strconv.ParseFloat(n.Pred, 64); err == nil {
						predictions[j][i] = val
					} else {
						predictions[j][i] = math.NaN()
					}
				}
			}
		}, fm, cases, 0)
	}

	return predictions
}

// Prediction consists of a predicted Value and it's associated variance
type Prediction struct {
	Value    float64
	Variance float64
}

// JackKnife estimates the variance of the predicted value from the RandomForest
// using an infinitesimal jackknife estimator for the variance.
//
// predictionSlice is a list of the complete predictions for each observation
// inbag is the inbag samples used by the RandomForest
//
// returns a Prediction for each value in the predictionSlice
func JackKnife(predictionSlice, inbag [][]float64) ([]*Prediction, error) {
	if len(predictionSlice) == 0 || len(inbag) == 0 {
		return nil, fmt.Errorf("prediction and inbag size must be equal")
	}

	var (
		m  = len(predictionSlice)
		n  = len(inbag)
		B  = float64(len(predictionSlice[0]))
		B2 = B * B
	)

	var (
		avgPreds = make([]float64, m)
		sums     = make([]float64, m)
		nvars    = make([]float64, n)
		output   = make([]*Prediction, m)
		wg       sync.WaitGroup
	)

	// normalize the prediction slices
	for i, predictions := range predictionSlice {
		wg.Add(1)
		go func(idx int, p []float64) {
			defer wg.Done()

			preds := govector.Vector(p)
			avgPred := preds.Mean()

			predictionSlice[idx] = preds.Apply(func(f float64) float64 { return f - avgPred })
			avgPreds[idx] = avgPred
		}(i, predictions)
	}
	wg.Wait()

	// calculate the raw infinitesimal jackknife values
	// V_{i, j} = \sum_{i, n} Cov( inbag[i], pred ) ^ 2
	for i, preds := range predictionSlice {
		wg.Add(1)
		go func(idx int, p []float64) {
			defer wg.Done()
			sum := 0.0
			for i := 0; i < n; i++ {
				val, err := govector.DotProduct(govector.Vector(inbag[i]), p)
				if err != nil {
					fmt.Printf("err!!! %v\n", err)
					continue
				}

				val *= val
				val /= B2
				sum += val
			}
			sums[idx] = sum
		}(i, preds)
	}
	wg.Wait()

	// normalize the IJ value with Monte-Carlo bias correction
	for i := 0; i < n; i++ {
		nvars[i] = govector.Vector(inbag[i]).Variance()
	}
	nvar := govector.Vector(nvars).Mean()

	for idx, preds := range predictionSlice {
		bv := 0.0
		for i := 0; i < len(preds); i++ {
			bv += preds[i] * preds[i]
		}

		output[idx] = &Prediction{
			Value:    avgPreds[idx],
			Variance: sums[idx] - float64(n)*nvar*(bv/B)/B,
		}

	}

	return output, nil
}
