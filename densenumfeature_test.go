package CloudForest

import "testing"

func TestNumFeature(t *testing.T) {

	name := "numfeature"

	f := &DenseNumFeature{
		make([]float64, 0, 0),
		make([]bool, 0, 0),
		name,
		false}

	f.Append("0.1")
	f.Append("10.1")
	f.Append("10.2")

	if x := f.NCats(); x != 0 {
		t.Errorf("Numerical NCats = %(v) != 0", x)
	}

	codedSplit := 0.5
	cases := []int{0, 1, 2}

	l, r, m := f.Split(codedSplit, cases)
	if len(l) != 1 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Numerical Split Left, Right, Missing Lengths = %v %v %v not 1 2 0", len(l), len(r), len(m))
	}

	decodedsplit := f.DecodeSplit(codedSplit)

	fm := FeatureMatrix{[]Feature{f},
		map[string]int{name: 0},
		[]string{name}}

	if !f.GoesLeft(0, decodedsplit) {
		t.Errorf("Value %v sent right by spliter decoded from %v", f.NumData[0], codedSplit)
	}
	if f.GoesLeft(1, decodedsplit) {
		t.Errorf("Value %v sent left by spliter decoded from %v", f.NumData[1], codedSplit)
	}

	l, r, m = decodedsplit.Split(&fm, cases)

	if len(l) != 1 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Decoded Numerical Split Left, Right, Missing Lengths = %v %v %v not 1 2 0", len(l), len(r), len(m))
	}

	f.Append("0.0")
	cases = append(cases, 3)

	f.Append("0.0")
	cases = append(cases, 4)

	l, r, m = decodedsplit.Split(&fm, cases)

	if len(l) != 3 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Decoded Numerical Split Left, Right, Missing Lengths = %v %v %v not 3 2 0", len(l), len(r), len(m))
	}

	l, r, m = f.Split(codedSplit, cases)
	if len(l) != 3 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Numerical Split Left, Right, Missing Lengths = %v %v %v not 3 2 0", len(l), len(r), len(m))
	}

	//check self slitting

	allocs := NewBestSplitAllocs(5, f)

	_, split, _ := fm.BestSplitter(f, &cases, &[]int{0}, nil, 1, false, false, allocs)
	//fm.BestSplitter(target, cases, candidates, oob, leafSize, vet, evaloob, allocs)

	if split.(float64) != 0.1 {
		t.Errorf("Numerical feature didn't self split correctelly. Returned %v not 0.1", split)
	}

	l, r, m = f.Split(split, cases)
	if len(l) != 3 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Numerical Split Left, Right, Missing Lengths = %v %v %v not 3 2 0", len(l), len(r), len(m))
	}

	//and with a run of equals
	f.Append(".1")
	cases = append(cases, 5)
	f.Append(".1")
	cases = append(cases, 6)

	allocs = NewBestSplitAllocs(7, f)

	_, split, _ = fm.BestSplitter(f, &cases, &[]int{0}, nil, 1, false, false, allocs)
	//fm.BestSplitter(target, cases, candidates, oob, leafSize, vet, evaloob, allocs)

	if split.(float64) != 0.1 {
		t.Errorf("Numerical feature didn't self split correctelly with equal run. Returned %v not 0.1", split)
	}

	l, r, m = f.Split(split, cases)
	if len(l) != 5 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Numerical Split with equal run Left, Right, Missing Lengths = %v %v %v not 5 2 0", len(l), len(r), len(m))
	}

	//spliting between two runs of equals
	f.Append("10.1")
	cases = append(cases, 7)

	allocs = NewBestSplitAllocs(8, f)

	_, split, _ = fm.BestSplitter(f, &cases, &[]int{0}, nil, 1, false, false, allocs)
	//fm.BestSplitter(target, cases, candidates, oob, leafSize, vet, evaloob, allocs)

	sorted := true
	for i := 1; i < len(cases); i++ {
		if f.NumData[cases[i]] < f.NumData[cases[i-1]] {
			sorted = false
		}

	}
	if !sorted {
		t.Error("Numerical feature didn't sort cases.")
	}

	if split.(float64) != 0.1 {
		t.Errorf("Numerical feature didn't self split correctelly between equal runs. Returned %v not 0.1", split)
	}

	l, r, m = f.Split(split, cases)
	if len(l) != 5 || len(r) != 3 || len(m) != 0 {
		t.Errorf("After Coded Numerical Split between equal runs Left, Right, Missing Lengths = %v %v %v not 5 3 0", len(l), len(r), len(m))
	}

}
