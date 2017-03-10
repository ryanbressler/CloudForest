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

//Forest represents a collection of decision trees grown to predict the Target
type Forest struct {
	Target    string
	Trees     []*Tree
	Intercept float64
}

func (f *Forest) Copy() *Forest {
	if f == nil {
		return nil
	}

	trees := make([]*Tree, len(f.Trees))
	for i, tree := range f.Trees {
		trees[i] = tree.Copy()
	}

	return &Forest{
		Target:    f.Target,
		Intercept: f.Intercept,
		Trees:     trees,
	}
}

// ForestModel is a complete view of the RandomForest
type ForestModel struct {
	Forest      *Forest         // The underlying Rf
	InBag       [][]float64     // InBag samples used by the RF
	Predictions [][]float64     // Predicted values of te input data based on OOB samples
	Importance  *[]*RunningMean // Variable Importance for the RF
}

// ForestConfig is used to configure and tune the RandomForest
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
		importance = NewRunningMeans(len(fm.Data))
		nCases     = fm.Data[0].Length()
		trees      = make([]*Tree, config.NTrees)
		inbag      = make([][]float64, config.NSamples)
		candidates = make([]int, 0, len(fm.Data)-1)
	)

	// construct list of candidate features
	for i, feature := range fm.Data {
		if feature.GetName() != targetName {
			candidates = append(candidates, i)
		}
	}

	// instantiate inbag matrix
	if config.InBag {
		for i := range inbag {
			inbag[i] = make([]float64, config.NTrees)
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
			for _, c := range cases {
				inbag[c][i]++
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

func (f *Forest) Predict(fm *FeatureMatrix) []float64 {
	n := fm.Data[0].Length()
	bb := NewNumBallotBox(n)
	for _, tree := range f.Trees {
		tree.Vote(fm, bb)
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		pred, _ := strconv.ParseFloat(bb.Tally(i), 64)
		preds[i] = pred
	}
	return preds
}

func (f *Forest) PredictCat(fm *FeatureMatrix) []string {
	n := fm.Data[0].Length()

	bb := NewCatBallotBox(n)
	for _, tree := range f.Trees {
		tree.Vote(fm, bb)
	}

	preds := make([]string, n)
	for i := 0; i < n; i++ {
		preds[i] = bb.Tally(i)
	}
	return preds
}

// Prediction consists of a predicted Value and it's associated variance
type Prediction struct {
	Value    float64
	Variance float64
}

// JackKnife estimates the variance of the predicted values from the RandomForest
// using an infinitesimal jackknife estimator for the variance, given the
// in-bag samples used by the RandomForest, and the predicted values from each tree.
//
// The underlying algorithm is described in detail in the paper:
// "Confidence Intervals for Random Forests: The JackKnife and the Infinitesimal JackKnife"
//  by Wager S., et al. http://arxiv.org/pdf/1311.4555.pdf
//
// The complete R implementation is available here: https://github.com/swager/randomForestCI
//
func JackKnife(predictionSlice, inbag [][]float64) ([]*Prediction, error) {
	if len(predictionSlice) == 0 || len(inbag) == 0 {
		return nil, fmt.Errorf("prediction and inbag size must be equal")
	}

	var (
		m    = len(predictionSlice)
		n    = len(inbag)
		B    = float64(len(predictionSlice[0]))
		B2   = B * B
		nvar = avgVar(inbag)
	)

	var (
		avgPreds = make([]float64, m)
		sums     = make([]float64, m)
		output   = make([]*Prediction, m)
		errs     = make(chan error, m)
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
					errs <- err
					return
				}

				val *= val
				val /= B2
				sum += val
			}
			sums[idx] = sum
		}(i, preds)
	}
	wg.Wait()

	// check for errors
	if len(errs) > 0 {
		return nil, <-errs
	}

	// normalize the IJ value with Monte-Carlo bias correction
	for i, preds := range predictionSlice {
		wg.Add(1)
		go func(idx int, p []float64) {
			defer wg.Done()
			bv := 0.0
			for i := 0; i < len(p); i++ {
				bv += p[i] * p[i]
			}

			variance := sums[idx] - float64(n)*nvar*(bv/B)/B

			output[idx] = &Prediction{
				Value:    avgPreds[idx],
				Variance: variance,
			}
		}(i, preds)
	}
	wg.Wait()

	return output, nil
}

func avgVar(val [][]float64) float64 {
	vars := make([]float64, len(val))
	for i := 0; i < len(val); i++ {
		vars[i] = govector.Vector(val[i]).Variance()
	}
	return meanSlice(vars)
}

const maxSplits = 51

// The Forest.Predict function is a Predictor
// Since the PartialDependecyPlot can be used to interpret the results of
// any tree-based model, diagnostic functions such as PDP should use the Predictor type
// to avoid CloudForest.Forest specific context
type Predictor func(*FeatureMatrix) []float64

// PDP calculates the partial dependency of 1 or 2 classes for a given Predictor/Feature Matrix
// NOTE: this is currently only functional for regression trees, not classification trees
func PDP(f Predictor, data *FeatureMatrix, classes ...string) ([][]float64, error) {
	numClasses := len(classes)
	switch numClasses {
	case 0:
		return nil, fmt.Errorf("must provide at least one class")
	case 1:
		return singlePDP(f, data, classes[0]), nil
	case 2:
		return doublePDP(f, data, classes[0], classes[1]), nil
	default:
		return nil, fmt.Errorf("too many classes provided")
	}
}

// singlePDP calculates the partial dependency of a single class
func singlePDP(f Predictor, data *FeatureMatrix, class string) [][]float64 {
	xv, idx, ok := toSeq(data, class)
	if !ok {
		return nil
	}

	xData := data
	n := data.Data[0].Length()
	output := make([][]float64, len(xv.values))

	// reset the feature matrix
	old := xData.Data[idx]
	defer func() {
		data.Data[idx] = old
	}()

	// find the mean prediction for each value in the grid
	for i, val := range xv.values {
		xData.Data[idx] = &DenseNumFeature{
			NumData: rep(val, n),
			Missing: make([]bool, n),
			Name:    xv.name,
		}

		output[i] = []float64{val, meanSlice(f(xData))}
	}

	return output
}

// doublePDP calculates the partial dependency of two classes
func doublePDP(f Predictor, data *FeatureMatrix, classA, classB string) [][]float64 {
	xv, xIdx, ok := toSeq(data, classA)
	if !ok {
		return nil
	}

	yv, yIdx, ok := toSeq(data, classB)
	if !ok {
		return nil
	}

	xData := data
	n := data.Data[0].Length()
	output := make([][]float64, len(xv.values)*len(yv.values))
	i := 0

	// reset the feature matrix
	oldX := xData.Data[xIdx]
	oldY := xData.Data[yIdx]
	defer func() {
		data.Data[xIdx] = oldX
		data.Data[yIdx] = oldY
	}()

	// find the mean prediction for each value in the 2x2 grid
	for _, valX := range xv.values {
		for _, valY := range yv.values {
			xData.Data[yIdx] = &DenseNumFeature{
				NumData: rep(valY, n),
				Missing: make([]bool, n),
				Name:    yv.name,
			}

			xData.Data[xIdx] = &DenseNumFeature{
				NumData: rep(valX, n),
				Missing: make([]bool, n),
				Name:    xv.name,
			}

			output[i] = []float64{valX, valY, meanSlice(f(xData))}
			i++
		}
	}
	return output
}

type featureSeq struct {
	name   string
	values []float64
}

// toSeq converts returns a sequence of values for a given feature in the FM
func toSeq(data *FeatureMatrix, class string) (*featureSeq, int, bool) {
	idx, ok := data.Map[class]
	if !ok {
		return nil, 0, false
	}

	xv, ok := data.Data[idx].(*DenseNumFeature)
	if !ok {
		return nil, 0, false
	}

	n := xv.Length()
	vals := make([]float64, n)
	uniq := make(map[string]struct{})
	for i := 0; i < xv.Length(); i++ {
		str := xv.GetStr(i)
		val, err := strconv.ParseFloat(str, 64)
		if err != nil {
			continue
		}
		vals[i] = val
		uniq[str] = struct{}{}
	}

	nPts := minInt(len(uniq), maxSplits)
	xPt := seq(minSlice(vals), maxSlice(vals), nPts)
	return &featureSeq{xv.GetName(), xPt}, idx, true
}

func meanSlice(x []float64) float64 {
	return govector.Vector(x).Mean()
}

func minSlice(x []float64) float64 {
	return govector.Vector(x).Min()
}

func maxSlice(x []float64) float64 {
	return govector.Vector(x).Max()
}

func rep(val float64, n int) []float64 {
	output := make([]float64, n)
	for i := range output {
		output[i] = val
	}
	return output
}

func seq(start, end float64, n int) []float64 {
	output := make([]float64, n)
	if start > end || n == 0 {
		return output
	}

	step := (end - start) / float64(n-1)
	for i := range output {
		output[i] = start + (step * float64(i))
	}
	return output
}

func minInt(x, y int) int {
	if x < y {
		return x
	}
	return y
}
