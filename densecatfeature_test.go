package CloudForest

import (
	"fmt"
	"testing"
)

func TestCatFeature(t *testing.T) {

	//Start with a small cat feature and do some simple spliting tests
	//then build it up and do some best split finding tests

	name := "catfeature"

	f := &DenseCatFeature{
		&CatMap{make(map[string]int, 0),
			make([]string, 0, 0)},
		make([]int, 0, 0),
		make([]bool, 0, 0),
		name,
		false,
		false}

	fm := FeatureMatrix{[]Feature{f},
		map[string]int{name: 0},
		[]string{name}}

	f.Append("0")
	f.Append("1")
	f.Append("1")

	//f has 0 1 1

	if x := f.NCats(); x != 2 {
		t.Errorf("Boolean NCats = %v != 2", x)
	}

	fns := f.EncodeToNum()
	fn := fns[0].(NumFeature)

	if len(fns) != 1 || fn.Get(0) != 0.0 || fn.Get(1) != 1.0 || fn.Get(2) != 1.0 {
		t.Errorf("Error: cat feature %v encoded to %v", f.CatData, fn.(*DenseNumFeature).NumData)
	}

	codedSplit := 1
	cases := []int{0, 1, 2}

	l, r, m := f.Split(0, cases)
	if len(l) != 0 || len(r) != 3 || len(m) != 0 {
		t.Errorf("After Coded Boolean Split 0 Left, Right, Missing Lengths = %v %v %v not 0 3 0", len(l), len(r), len(m))
	}

	decodedsplit := f.DecodeSplit(0)

	l, r, m = decodedsplit.Split(&fm, cases)

	if len(l) != 0 || len(r) != 3 || len(m) != 0 {
		t.Errorf("After Decoded Boolean Split 0 Left, Right, Missing Lengths = %v %v %v not 0 3 0", len(l), len(r), len(m))
	}

	l, r, m = f.Split(1, cases)
	if len(l) != 1 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Boolean Split 1 Left, Right, Missing Lengths = %v %v %v not 1 2 0", len(l), len(r), len(m))
	}

	l, r, m = f.Split(2, cases)
	if len(l) != 2 || len(r) != 1 || len(m) != 0 {
		t.Errorf("After Coded Boolean Split  2 Left, Right, Missing Lengths = %v %v %v not 2 1 0", len(l), len(r), len(m))
	}

	decodedsplit = f.DecodeSplit(codedSplit)

	l, r, m = decodedsplit.Split(&fm, cases)

	if len(l) != 1 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Decoded Boolean Split Left, Right, Missing Lengths = %v %v %v not 1 2 0", len(l), len(r), len(m))
	}

	f.Append("0")
	cases = append(cases, 3)
	// f has 0 1 1 0

	l, r, m = decodedsplit.Split(&fm, cases)

	if len(l) != 2 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Decoded Boolean Split Left, Right, Missing Lengths = %v %v %v not 2 2 0", len(l), len(r), len(m))
	}

	l, r, m = f.Split(codedSplit, cases)
	if len(l) != 2 || len(r) != 2 || len(m) != 0 {
		t.Errorf("After Coded Boolean Split Left, Right, Missing Lengths = %v %v %v not 2 2 0", len(l), len(r), len(m))
	}

	f.Append("0")
	cases = append(cases, 4)

	allocs := NewBestSplitAllocs(5, f)

	_, split, _, _ := fm.BestSplitter(f, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	if split.(int) != 1 {
		t.Errorf("Boolean feature didn't self split. Returned %v", split)
	}

	//f has 0 1 1 0 0

	target := f.Copy()
	target.Append("1")
	f.Append("NA")

	//f has 0 1 1 0 0 NA
	//target has 0 1 1 0 0 1

	if f.IsMissing(5) != true || f.MissingVals() != true || f.HasMissing != true {
		t.Error("Feature with missing values claims not")
	}

	if target.IsMissing(5) == true || target.MissingVals() == true || target.(*DenseCatFeature).HasMissing == true {
		t.Error("Target with missing values claims not")
	}

	cases = append(cases, 5)

	allocs = NewBestSplitAllocs(6, target)

	_, split, _, _ = fm.BestSplitter(target, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	if split.(int) != 1 {
		t.Errorf("Boolean with missing val feature didn't split non missing copy. Returned %v", split)
	}

	target.PutStr(5, "2")

	//f has 0 1 1 0 0 NA
	//target has 0 1 1 0 0 2

	allocs = NewBestSplitAllocs(6, target)

	_, split, _, _ = fm.BestSplitter(target, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	if split.(int) != 1 {
		t.Errorf("Trinary target with bool missing val feature didn't split. Returned %v", split)
	}

	f.PutStr(5, "0")

	//f has 0 1 1 0 0 0
	//target has 0 1 1 0 0 2
	// zero should go left by itself, code = 1

	allocs = NewBestSplitAllocs(6, target)

	_, split, _, _ = fm.BestSplitter(target, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	if split.(int) != 1 {
		t.Errorf("Trinary target with bool missing val feature didn't split. Returned %v", split)
	}

	mediumf := &DenseCatFeature{
		&CatMap{make(map[string]int, 0),
			make([]string, 0, 0)},
		make([]int, 0, 0),
		make([]bool, 0, 0),
		"mediumf",
		false,
		false}

	for i := 0; i < 6; i++ {
		mediumf.Append(fmt.Sprintf("%v", i))
	}

	mediumfm := FeatureMatrix{[]Feature{mediumf},
		map[string]int{mediumf.Name: 0},
		[]string{mediumf.Name}}

	//f has 0 1 1 0 0 0
	//target has 0 1 1 0 0 2
	//medieumf has 0 1 2 3 4 5

	allocs = NewBestSplitAllocs(6, target)

	//split f by medium f should send 1 and 2 to one side, coded 6
	_, split, _, _ = mediumfm.BestSplitter(f, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	if split.(int) != 6 {
		t.Errorf("Binary target with 6 valued feature didn't split. Returned %v", split)
	}

	l, r, m = mediumf.Split(split, cases)
	if len(l) != 2 || len(r) != 4 || len(m) != 0 {
		t.Errorf("After Coded Boolean vs Multivalued Split Left, Right, Missing Lengths = %v %v %v not 2 4 0", len(l), len(r), len(m))
	}

	decodedsplit = mediumf.DecodeSplit(split)

	l, r, m = decodedsplit.Split(&mediumfm, cases)
	//fmt.Println(decodedsplit.Left)

	if len(l) != 2 || len(r) != 4 || len(m) != 0 {
		t.Errorf("After Decoded Boolean Split Left, Right, Missing Lengths = %v %v %v not 2 4 0", len(l), len(r), len(m))
	}

	//target.Append(v)

}

func TestBigCatFeature(t *testing.T) {

	bigf := &DenseCatFeature{
		&CatMap{make(map[string]int, 0),
			make([]string, 0, 0)},
		make([]int, 0, 0),
		make([]bool, 0, 0),
		"big",
		false,
		false}

	boolf := &DenseCatFeature{
		&CatMap{make(map[string]int, 0),
			make([]string, 0, 0)},
		make([]int, 0, 0),
		make([]bool, 0, 0),
		"bool",
		false,
		false}

	cases := make([]int, 40, 40)
	for i := 0; i < 40; i++ {
		bigf.Append(fmt.Sprintf("%v", i))
		boolf.Append(fmt.Sprintf("%v", i < 20))
		cases[i] = i
	}

	bigfm := FeatureMatrix{[]Feature{bigf},
		map[string]int{bigf.Name: 0},
		[]string{bigf.Name}}

	allocs := NewBestSplitAllocs(40, boolf)

	//split f by medium f should send 1 and 2 to one side, coded 6
	_, split, _, _ := bigfm.BestSplitter(boolf, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	l, r, m := bigf.Split(split, cases)
	if len(l) != 20 || len(r) != 20 || len(m) != 0 {
		t.Errorf("After Coded big split Left, Right, Missing Lengths = %v %v %v not 20 20 0", len(l), len(r), len(m))
	}

	decodedsplit := bigf.DecodeSplit(split)

	l, r, m = decodedsplit.Split(&bigfm, cases)
	//fmt.Println(decodedsplit.Left)

	if len(l) != 20 || len(r) != 20 || len(m) != 0 {
		t.Errorf("After Decoded big split Left, Right, Missing Lengths = %v %v %v not 20 20 0", len(l), len(r), len(m))
	}

	bigf.RandomSearch = true
	_, split, _, _ = bigfm.BestSplitter(boolf, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	l, r, m = bigf.Split(split, cases)
	//won't perfectelly split but should do okay
	if len(l) < 18 || len(r) < 18 || len(m) != 0 {
		t.Errorf("After Coded big random split Left, Right, Missing Lengths = %v %v %v not >=18 >=18 0", len(l), len(r), len(m))
	}

	bigf.PutMissing(23)
	bigf.RandomSearch = false

	//split f by medium f should send 1 and 2 to one side, coded 6
	_, split, _, _ = bigfm.BestSplitter(boolf, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	l, r, m = bigf.Split(split, cases)
	if len(l) < 19 || len(r) < 19 || len(m) != 1 {
		t.Errorf("After Coded big missing split Left, Right, Missing Lengths = %v %v %v not 19 20 1", len(l), len(r), len(m))
	}

	decodedsplit = bigf.DecodeSplit(split)

	l, r, m = decodedsplit.Split(&bigfm, cases)
	//fmt.Println(decodedsplit.Left)

	if len(l) < 19 || len(r) < 19 || len(m) != 1 {
		t.Errorf("After Decoded big split Left, Right, Missing Lengths = %v %v %v not 19 20 1", len(l), len(r), len(m))
	}

	bigf.RandomSearch = true
	_, split, _, _ = bigfm.BestSplitter(boolf, &cases, &[]int{0}, 1, nil, 1, false, false, false, false, allocs, 0)

	l, r, m = bigf.Split(split, cases)
	//won't perfectelly split but should do okay
	if len(l) < 18 || len(r) < 18 || len(m) != 1 {
		t.Errorf("After Coded big random split Left, Right, Missing Lengths = %v %v %v not >=18 >=18 1", len(l), len(r), len(m))
	}

}
