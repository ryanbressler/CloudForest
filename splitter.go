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
	Right     map[string]bool
}

//func

//Splitter.Split seperates cases []int using the data in fm *FeatureMatrix
//and returns left and right []ints.
//It applies either a Numerical or Catagorical split. In the Numerical case
//everything <= to Value is sent left; for the Catagorical case a look up 
//table is used.
func (s *Splitter) Split(fm *FeatureMatrix, cases []int) (l []int, r []int) {
	l = make([]int, 0)
	r = make([]int, 0)

	f := fm.Data[fm.Map[s.Feature]]

	switch s.Numerical {
	case true:
		s.SplitNum(&f, &cases, &l, &r)
	case false:
		s.SplitCat(&f, &cases, &l, &r)
	}

	return
}

func (s *Splitter) DescribeMap(input map[string]bool) string {
	keys := make([]string, 0)
	for k := range input {
		keys = append(keys, k)
	}
	return "\"" + strings.Join(keys, ":") + "\""
}

func (s *Splitter) SplitNum(f *Feature, cases *[]int, l *[]int, r *[]int) {
	for _, i := range *cases {
		if f.Missing[i] == false {
			switch {
			case f.NumData[i] <= s.Value:
				*l = append(*l, i)
			default:
				*r = append(*r, i)
			}
		}

	}
	return
}

func (s *Splitter) SplitCat(f *Feature, cases *[]int, l *[]int, r *[]int) {
	for _, i := range *cases {
		if f.Missing[i] == false {

			v := f.Back[f.CatData[i]]
			switch {
			case s.Left[v]:
				*l = append(*l, i)
			default:
				*r = append(*r, i)
			}
		}

	}
	return
}
