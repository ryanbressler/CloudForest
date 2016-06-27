package CloudForest

//"fmt"

//Splitter contains fields that can be used to cases by a single feature. The split
//can be either numerical in which case it is defined by the Value field or
//categorical in which case it is defined by the Left and Right fields.
type Splitter struct {
	Feature   string
	Numerical bool
	Value     float64
	Left      map[string]bool
}

func (c *Splitter) Copy() *Splitter {
	if c == nil {
		return nil
	}

	lft := make(map[string]bool)
	if c.Left != nil {
		for s, b := range c.Left {
			lft[s] = b
		}
	}

	return &Splitter{
		Feature:   c.Feature,
		Numerical: c.Numerical,
		Value:     c.Value,
		Left:      lft,
	}
}

//func

/*
Split splits a slice of cases into left, right and missing slices without allocating
a new underlying array by sorting cases into left, missing, right order and returning
slices that point to the left and right cases.
*/
func (s *Splitter) Split(fm *FeatureMatrix, cases []int) (l []int, r []int, m []int) {
	length := len(cases)

	lastleft := -1
	lastright := length
	swaper := 0

	f := fm.Data[fm.Map[s.Feature]]

	//Move left cases to the start and right cases to the end so that missing cases end up
	//in between.
	hasmissing := f.MissingVals()
	for i := 0; i < lastright; i++ {
		if hasmissing && f.IsMissing(cases[i]) {
			continue
		}
		if f.GoesLeft(cases[i], s) {
			lastleft++
			if i != lastleft {

				swaper = cases[i]
				cases[i] = cases[lastleft]
				cases[lastleft] = swaper
				i--

			}

		} else {
			//Right
			lastright--
			swaper = cases[i]
			cases[i] = cases[lastright]
			cases[lastright] = swaper
			i--

		}

	}
	//fmt.Println(cases, lastleft, lastright)
	l = cases[:lastleft+1]
	r = cases[lastright:]
	m = cases[lastleft+1 : lastright]

	return
}
