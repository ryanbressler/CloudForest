package CloudForest

import (
	"strings"
	"testing"
)

func BenchmarkIris(b *testing.B) {

	candidates := []int{1, 2, 3, 4}
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
		tree.Grow(fm, target, cases, candidates, nil, 2, 1, 0, false, false, false, false, false, nil, nil, allocs)

	}
}

func BenchmarkBoston(b *testing.B) {

	boston := strings.NewReader(boston_housing)

	fm := ParseARFF(boston)

	candidates := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	target := fm.Data[fm.Map["class"]]

	cases := make([]int, 0, fm.Data[0].Length())
	for i := 0; i < fm.Data[0].Length(); i++ {
		cases = append(cases, i)
	}
	allocs := NewBestSplitAllocs(len(cases), target)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree := NewTree()
		tree.Grow(fm, target, cases, candidates, nil, 2, 1, 0, false, false, false, false, false, nil, nil, allocs)

	}
}

func BenchmarkBestNumSplit(b *testing.B) {

	// irisreader := strings.NewReader(irisarff)
	// fm := ParseARFF(irisreader)
	// targeti := 4

	irisreader := strings.NewReader(irislibsvm)
	fm := ParseLibSVM(irisreader)
	targeti := 0

	targetf := fm.Data[targeti]

	cases := make([]int, 0, 150)
	for i := 0; i < fm.Data[0].Length(); i++ {
		cases = append(cases, i)
	}
	allocs := NewBestSplitAllocs(len(cases), targetf)

	parentImp := targetf.Impurity(&cases, allocs.Counter)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = fm.Data[1].BestSplit(targetf, &cases, parentImp, 1, false, allocs)

	}
}
