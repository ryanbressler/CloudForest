package CloudForest

import (
	"strings"
)

//Splitter contains fields that can be used to cases by a single feature. The split
//can be either numerical in which case it is defined by the Value field or
//catagorical in which case it is defined by the Left and Right fields.
type Splitter struct {
	Feature   string
	Numerical bool
	Value     float64
	Left      map[string]bool
}

//func

//Splitter.Split seperates cases []int using the data in fm *FeatureMatrix
//and returns left and right []ints.
//It applies either a Numerical or Catagorical split. In the Numerical case
//everything <= to Value is sent left; for the Catagorical case a look up
//table is used.
func (s *Splitter) Split(fm *FeatureMatrix, cases []int) (l []int, r []int) {
	length := len(cases)
	l = make([]int, 0, length)
	r = make([]int, 0, length)

	f := fm.Data[fm.Map[s.Feature]]

	switch s.Numerical {
	case true:
		s.SplitNum(&f, &cases, &l, &r)
	case false:
		s.SplitCat(&f, &cases, &l, &r)
	}

	return
}

/*
SplitInPlace splits a slice of cases into left and right slices without allocating
a new underlying array by sorting cases into left,missing,right order and returning
slices that point to the left and right cases.
*/
func (s *Splitter) SplitInPlace(fm *FeatureMatrix, cases []int) (l []int, r []int) {
	length := len(cases)

	lastleft := -1
	lastright := length
	swaper := 0

	f := fm.Data[fm.Map[s.Feature]]

	for i := 0; i < lastright; i++ {
		if f.Missing[cases[i]] {
			continue
		}
		if (s.Numerical && f.NumData[cases[i]] <= s.Value) ||
			(!s.Numerical && s.Left[f.Back[f.CatData[cases[i]]]]) {
			//Left
			if i != lastleft+1 {
				lastleft += 1
				swaper = cases[i]
				cases[i] = cases[lastleft]
				cases[lastleft] = swaper
				i--

			}

		} else {
			//Right
			lastright -= 1
			swaper = cases[i]
			cases[i] = cases[lastright]
			cases[lastright] = swaper
			i -= 1

		}

	}

	l = cases[:lastleft+1]
	r = cases[lastright:]

	return
}

func (s *Splitter) DescribeMap(input map[string]bool) string {
	keys := make([]string, 0)
	for k := range input {
		keys = append(keys, k)
	}
	return "\"" + strings.Join(keys, ":") + "\""
}

//SplitNum is a low level fuction that splits the supplied cases into the supplied
//left and right *[]ints which should be empty when SplitNum is called.
func (s *Splitter) SplitNum(f *Feature, cases *[]int, l *[]int, r *[]int) {
	for _, i := range *cases {
		if f.Missing[i] == false {
			if f.NumData[i] <= s.Value {
				*l = append(*l, i)
			} else {
				*r = append(*r, i)
			}
		}

	}
	return
}

func (s *Splitter) SplitCat(f *Feature, cases *[]int, l *[]int, r *[]int) {
	/*BUG SplitCat is much slower then the splitting used in the search
	because it uses map[string]bool representations of the left and right instead
	of bits. This is for compatability with diffrent feature matrixes
	as the int string corospondance is not stable.*/
	v := ""
	for _, i := range *cases {

		if f.Missing[i] == false {
			v = f.Back[f.CatData[i]]
			if s.Left[v] {
				*l = append(*l, i)
			} else {
				*r = append(*r, i)
			}
		}

	}
	return
}
