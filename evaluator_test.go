package CloudForest

import (
	"fmt"
	"strings"
	"testing"

	"github.com/bmizerany/assert"
)

func setupCategorical() (*Forest, *FeatureMatrix) {
	irisreader := strings.NewReader(irislibsvm)
	fm := ParseLibSVM(irisreader)
	targeti := 0
	cattarget := fm.Data[targeti]
	config := &ForestConfig{
		NSamples: fm.Data[0].Length(),
		MTry:     3,
		NTrees:   10,
		LeafSize: 1,
	}

	sample := &FeatureMatrix{
		Data: make([]Feature, len(fm.Map)),
		Map:  make(map[string]int),
	}
	for k, v := range fm.Map {
		var feature Feature
		if v == 0 {
			feature = NewDenseCatFeature(k)
		} else {
			feature = NewDenseNumFeature(k)
		}
		sample.Map[k] = v
		sample.Data[v] = feature
		sample.Data[v].Append(fm.Data[v].GetStr(0))
	}

	model := GrowRandomForest(fm, cattarget, config)
	return model.Forest, sample
}

func setupNumeric() (*Forest, *FeatureMatrix) {
	boston := strings.NewReader(boston_housing)
	fm := ParseARFF(boston)
	target := fm.Data[fm.Map["class"]]
	sample := &FeatureMatrix{
		Data: make([]Feature, len(fm.Map)),
		Map:  make(map[string]int),
	}
	for k, v := range fm.Map {
		sample.Map[k] = v
		sample.Data[v] = NewDenseNumFeature(k)
		sample.Data[v].Append(fm.Data[v].GetStr(0))
	}
	config := &ForestConfig{
		NSamples: target.Length(),
		MTry:     4,
		NTrees:   20,
		LeafSize: 1,
		MaxDepth: 4,
		InBag:    true,
	}
	model := GrowRandomForest(fm, target, config)
	return model.Forest, sample
}

func roughlyEqual(t *testing.T, x, y float64) {
	assert.Equal(t, fmt.Sprintf("%.4f", x), fmt.Sprintf("%.4f", y))
}

func TestEvaluator(t *testing.T) {
	forest, sample := setupNumeric()
	predVal := forest.Predict(sample)[0]

	evalPW := NewPiecewiseFlatForest(forest)
	evalVal := evalPW.EvaluateNum(sample)[0]
	roughlyEqual(t, predVal, evalVal)

	evalCT := NewContiguousFlatForest(forest)
	evalVal = evalCT.EvaluateNum(sample)[0]
	roughlyEqual(t, predVal, evalVal)
}

func TestCatEvaluator(t *testing.T) {
	forest, sample := setupCategorical()
	pred := forest.PredictCat(sample)[0]

	pw := NewPiecewiseFlatForest(forest)
	predPW := pw.EvaluateCat(sample)[0]
	assert.Equal(t, pred, predPW)

	ct := NewContiguousFlatForest(forest)
	predCT := ct.EvaluateCat(sample)[0]
	assert.Equal(t, predPW, predCT)
}

// BenchmarkPredict-8            	  100000	     12542 ns/op
func BenchmarkPredict(b *testing.B) {
	forest, sample := setupNumeric()

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		forest.Predict(sample)
	}
	b.StopTimer()
}

// BenchmarkFlatForest-8         	 2000000	       821 ns/op
func BenchmarkFlatForest(b *testing.B) {
	forest, sample := setupNumeric()
	pw := NewPiecewiseFlatForest(forest)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		pw.EvaluateNum(sample)
	}
	b.StopTimer()
}

// BenchmarkContiguousForest-8   	 5000000	       339 ns/op
func BenchmarkContiguousForest(b *testing.B) {
	forest, sample := setupNumeric()
	ct := NewContiguousFlatForest(forest)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ct.EvaluateNum(sample)
	}
	b.StopTimer()
}
