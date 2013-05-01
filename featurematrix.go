package CloudForest

import (
	"encoding/csv"
	"io"
	"log"
	"math/big"
)

//FeatureMatrix contains a slice of Features and a Map to look of the index of a feature
//by its string id.
type FeatureMatrix struct {
	Data []Feature
	Map  map[string]int
}

/*
BestSplitter finds the best splitter from a number of canidate features to slit on by looping over
all features and calling BestSplit.

Pointers to slices for l and r are used to reduce realocations during repeated calls
and will not contain meaningfull results.

l and r should have capacity >=  cap(cases) to avoid resizing.
*/
func (fm *FeatureMatrix) BestSplitter(target *Feature,
	cases []int,
	canidates []int,
	itter bool,
	l *[]int,
	r *[]int) (s *Splitter, impurityDecrease float64) {

	impurityDecrease = minImp

	var f, bestF *Feature
	var num, bestNum, inerImp float64
	var cat, bestCat int
	var bigCat, bestBigCat *big.Int

	var counter []int
	sorter := new(SortableFeature)
	if target.Numerical == false {
		counter = make([]int, len(target.Back), len(target.Back))
	}

	parentImp := target.Impurity(&cases, &counter)

	left := *l
	right := *r

	for _, i := range canidates {
		left = left[:]
		right = right[:]
		f = &fm.Data[i]
		num, cat, bigCat, inerImp = f.BestSplit(target, &cases, parentImp, itter, &left, &right, &counter, sorter)
		//BUG more stringent cutoff in BestSplitter?
		if inerImp > minImp && inerImp > impurityDecrease {
			bestF = f
			impurityDecrease = inerImp
			bestNum = num
			bestCat = cat
			bestBigCat = bigCat
		}

	}
	if impurityDecrease > minImp {
		s = bestF.DecodeSplit(bestNum, bestCat, bestBigCat)
	}
	return
}

//Parse an AFM (anotated feature matrix) out of an io.Reader
//AFM format is a tsv with row and column headers where the row headers start with
//N: indicating numerical, C: indicating catagorical or B: indicating boolean
//For this parser features without N: are assumed to be catagorical
func ParseAFM(input io.Reader) *FeatureMatrix {
	data := make([]Feature, 0, 100)
	lookup := make(map[string]int, 0)
	tsv := csv.NewReader(input)
	tsv.Comma = '\t'
	_, err := tsv.Read()
	if err == io.EOF {
		return &FeatureMatrix{data, lookup}
	} else if err != nil {
		log.Print("Error:", err)
		return &FeatureMatrix{data, lookup}
	}

	count := 0
	for {
		record, err := tsv.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Print("Error:", err)
			break
		}
		data = append(data, ParseFeature(record))
		lookup[record[0]] = count
		count++
	}
	return &FeatureMatrix{data, lookup}
}
