package CloudForest

import (
	"strings"
	"testing"

	"github.com/bmizerany/assert"
)

func setup() (*Forest, *FeatureMatrix) {
	boston := strings.NewReader(boston_housing)
	fm := ParseARFF(boston)
	target := fm.Data[fm.Map["class"]]
	sample := &FeatureMatrix{
		Data: make([]Feature, len(fm.Map)),
		Map:  make(map[string]int),
	}
	for k, v := range fm.Map {
		sample.Map[k] = v
		sample.Data[v] = &DenseNumFeature{Name: k}
		sample.Data[v].Append(fm.Data[v].GetStr(0))
	}
	config := &ForestConfig{
		NSamples: target.Length(),
		MTry:     4,
		NTrees:   20,
		LeafSize: 1,
		InBag:    true,
	}
	model := GrowRandomForest(fm, target, config)
	return model.Forest, sample
}

func TestEvaluator(t *testing.T) {
	forest, sample := setup()

	predVal := forest.Predict(sample)[0]

	evalPW := NewPiecewiseFlatForest(forest)
	evalVal := evalPW.Evaluate(sample)[0]
	assert.Equal(t, predVal, evalVal)

	evalCT := NewContiguousFlatForest(forest)
	evalVal = evalCT.Evaluate(sample)[0]
	assert.Equal(t, predVal, evalVal)
}

// BenchmarkPredict-8            	    5000	    243381 ns/op
func BenchmarkPredict(b *testing.B) {
	forest, sample := setup()

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		forest.Predict(sample)
	}
	b.StopTimer()
}

// BenchmarkFlatForest-8         	  100000	     10060 ns/op
func BenchmarkFlatForest(b *testing.B) {
	forest, sample := setup()
	pw := NewPiecewiseFlatForest(forest)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		pw.Evaluate(sample)
	}
	b.StopTimer()
}

// BenchmarkContiguousForest-8   	  200000	      8397 ns/op
func BenchmarkContiguousForest(b *testing.B) {
	forest, sample := setup()
	ct := NewContiguousFlatForest(forest)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ct.Evaluate(sample)
	}
	b.StopTimer()
}
