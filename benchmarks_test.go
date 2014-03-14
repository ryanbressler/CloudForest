package CloudForest

import (
	"strings"
	"testing"
)

func BenchmarkIris(b *testing.B) {

	candidates := []int{0, 1, 2, 3}
	// irisreader := strings.NewReader(irisarff)
	// fm := ParseARFF(irisreader)
	// targeti := 4

	irisreader := strings.NewReader(irislibsvm)
	fm := ParseLibSVM(irisreader)
	targeti := 0

	target := fm.Data[targeti]

	cases := make([]int, 0, 150)
	for i := 0; i < fm.Data[0].Length(); i++ {
		cases = append(cases, i)
	}
	allocs := NewBestSplitAllocs(len(cases), target)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree := NewTree()
		tree.Grow(fm, target, cases, candidates, nil, 2, 1, false, false, false, nil, nil, allocs)

	}
}
