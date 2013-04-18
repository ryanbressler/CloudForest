package CloudForest

import ()

//Splitter contains fields that can be used to cases by a single feature. The split
//can be either numerical in which case it is defined by the Value field or 
//catagorical in which case it is defined by the Left and Right fields.
type Splitter struct {
	Feature   string
	Numerical bool
	Value     Num
	Left      map[string]bool
	Right     map[string]bool
}

//Splitter.Split seperates cases []int using the data in fm *FeatureMatrix
//and returns left and right []ints.
//It applies either a Numerical or Catagorical split. In the Numerical case
//everything <= to Value is sent left; for the Catagorical case a look up 
//table is used.
func (s *Splitter) Split(fm *FeatureMatrix, cases []int) ([]int, []int) {
	l := make([]int, 0)
	r := make([]int, 0)
	f := fm.Data[fm.Map[s.Feature]]

	switch s.Numerical {
	case true:
		for _, i := range cases {
			if f.Missing[i] == false {
				switch {
				case f.Data[i] <= s.Value:
					l = append(l, i)
				default:
					r = append(r, i)
				}
			}

		}
	case false:
		for _, i := range cases {
			if f.Missing[i] == false {

				v := f.Back[f.Data[i]]
				switch {
				case s.Left[v]:
					l = append(l, i)
				case s.Right[v]:
					r = append(r, i)
				}
			}

		}
	}

	return l, r
}
