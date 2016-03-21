package CloudForest

import (
	"bytes"
	"strings"
	"testing"
)

//A toy feature matrix where either of the first
//two variables should be easilly predictible
//by the other by a single greedy tree.
var constantsfm = `.	0	1	2	3	4	5	6	7
C:CatTarget	0	0	0	0	0	1	1	1
N:GoodVals	0	0	0	0	0	1	1	1
C:Const1	0	0	0	0	0	0	0	1
C:Const2	0	0	0	0	0	0	0	1
C:Const3	0	0	0	0	0	0	0	1
N:Const4	0	0	0	0	0	0	0	1
N:Const5	0	0	0	0	0	0	0	1
N:Const6	0	0	0	0	0	0	0	1`

func TestBestSplitter(t *testing.T) {
	fm := readFm()

	target := fm.Data[0]
	cases := &[]int{0, 1, 2, 3, 4, 5, 6}
	candidates := []int{1, 2, 3, 4, 5, 6, 7}
	allocs := NewBestSplitAllocs(len(*cases), target)

	_, imp, constant := fm.Data[1].BestSplit(target, cases, 1, 1, false, allocs)
	if imp <= minImp || constant == true {
		t.Errorf("Good feature had imp %v and constant: %v", imp, constant)
	}

	_, imp, constant = fm.Data[2].BestSplit(target, cases, 1, 1, false, allocs)
	if imp > minImp || constant == false {
		t.Errorf("Constant cat feature had imp %v and constant: %v %v", imp, constant, fm.Data[2].(*DenseCatFeature).CatData)
	}

	_, imp, constant = fm.Data[7].BestSplit(target, cases, 1, 1, false, allocs)
	if imp > minImp || constant == false {
		t.Errorf("Constant num feature had imp %v and constant: %v", imp, constant)
	}

	fi, split, impDec, nconstants := fm.BestSplitter(target, cases, &candidates, len(candidates), nil, 1, true, false, false, false, allocs, 0)
	if fi != 1 || split == nil || impDec == minImp || nconstants != 6 {
		t.Errorf("BestSplitter couldn't find non constant feature and six constants fi: %v split: %v impDex: %v nconstants: %v ", fi, split, impDec, nconstants)
	}

	for i := 0; i < 7; i++ {

		candidates = []int{1, 2, 3, 4, 5, 6, 7}

		fi, split, impDec, nconstants = fm.BestSplitter(target, cases, &candidates, 1, nil, 1, true, false, false, false, allocs, i)
		if fi != 1 || split == nil || impDec == minImp {
			t.Errorf("BestSplitter couldn't find non constant feature with mTry=1 and %v known constants fi: %v split: %v impDex: %v nconstants: %v ", i, fi, split, impDec, nconstants)
		}

		candidates = []int{1, 2, 3, 4, 5, 6, 7}
		fi, split, impDec, nconstants = fm.BestSplitter(target, cases, &candidates, len(candidates), nil, 1, true, false, false, false, allocs, i)
		if fi != 1 || split == nil || impDec == minImp || nconstants != 6 {
			t.Errorf("BestSplitter couldn't find non constant feature and six constants with %v known constants fi: %v split: %v impDex: %v nconstants: %v ", i, fi, split, impDec, nconstants)
		}
	}
}

func TestFmWrite(t *testing.T) {
	fm := readFm()

	writer := &bytes.Buffer{}
	if err := fm.WriteFM(writer, "\t", true); err != nil {
		t.Fatalf("could not write feature matrix: %v", err)
	}

	if writer.String() == "" {
		t.Fatalf("could not write FM - buffer is empty")
	}
	firstLen := writer.Len()

	writer = &bytes.Buffer{}
	if err := fm.WriteFM(writer, "\t", false); err != nil {
		t.Fatalf("could not write feature matrix: %v", err)
	}

	if writer.String() == "" {
		t.Fatalf("could not write FM - buffer is empty")
	}
	secondLen := writer.Len()

	if firstLen != secondLen {
		t.Fatalf("expected buffers to have the same length: %v != %v", firstLen, secondLen)
	}
}

func readFm() *FeatureMatrix {
	fmReader := strings.NewReader(constantsfm)
	return ParseAFM(fmReader)
}
