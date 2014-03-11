package CloudForest

import (
	//"fmt"
	//"math"
	"github.com/ryanbressler/CloudForest/sortby"
	//"sort"
)

/*
Sortable feature is a wrapper for a feature and set of cases that satisfies the
sort.Interface interface so that the case indexes in Cases can be sorted using
sort.Sort
*/
type SortableFeature struct {
	//Feature NumFeature
	Vals  []float64
	Cases []int
}

//introsort + heapsort as in scikits learn's tree implementaion. For now go's sort is actually faster as it does less comparison
func (sf *SortableFeature) Sort() {
	//n := len(sf.Cases)
	// maxd := 2 * int(math.Log(float64(n)))
	// sf.introsort(0, n, maxd)
	sortby.SortBy(&sf.Cases, &sf.Vals)
	//sf.heapsort(0, n)
	//sort.Sort(sf)
}

//Len returns the number of cases.
func (sf *SortableFeature) Len() int {
	return len(sf.Cases)
}

//Less determines if the ith case is less then the jth case.
func (sf *SortableFeature) Less(i int, j int) bool {
	v := sf.Vals
	return v[i] < v[j]
	return sf.Vals[i] < sf.Vals[j]
	//return sf.Feature.Get(sf.Cases[i]) < sf.Feature.Get(sf.Cases[j])

}

//Swap exchanges the ith and jth cases.
func (sf *SortableFeature) Swap(i int, j int) {
	c := sf.Cases
	v := c[i]
	c[i] = c[j]
	c[j] = v
	vs := sf.Vals
	w := vs[i]
	vs[i] = vs[j]
	vs[j] = w

}

func (sf *SortableFeature) Load(vals *[]float64, cases *[]int) {
	sf.Cases = *cases
	for i, p := range *cases {
		sf.Vals[i] = (*vals)[p]
	}
}

func (sf *SortableFeature) median3(start int, end int) float64 {
	a := sf.Vals[start]
	b := sf.Vals[(start+end)/2]
	c := sf.Vals[end-1]
	switch {
	case a < b:
		switch {
		case b < c:
			return b
		case a < c:
			return c
		default:
			return a
		}
	case b < c:
		if a < c {
			return a
		} else {
			return c
		}
	default:
		return b
	}

}

func (sf *SortableFeature) introsort(start int, end int, maxd int) {
	var pivot float64
	var i, l, r int

	for (end - start) > 1 {
		if maxd <= 0 {
			//fmt.Println("heap sorting !! ", start, " ", end)
			sf.heapsort(start, end)
			return
		}

		maxd--
		pivot = sf.median3(start, end)
		//i = l = 0
		i = start
		l = start
		r = end
		for i < r {
			switch {
			case sf.Vals[i] < pivot:
				sf.Swap(i, l)
				i++
				l++
			case sf.Vals[i] >= pivot:
				r--
				sf.Swap(i, r)
			default:
				i++
			}

		}
		sf.introsort(start, l, maxd)
		start = r
	}

}

func (sf *SortableFeature) siftdown(start int, end int) {
	var child, maxind, root int
	root = start
	for {
		child = root*2 + 1
		maxind = root
		if child < end && sf.Vals[maxind] < sf.Vals[child] {
			maxind = child
		}
		if child+1 < end && sf.Vals[maxind] < sf.Vals[child+1] {
			maxind = child + 1
		}

		if maxind == root {
			return
		} else {
			sf.Swap(root, maxind)
			root = maxind
		}
	}
}

func (sf *SortableFeature) heapsort(s int, e int) {
	var start, end int
	start = (s + e - 2) / 2
	end = e
	for {
		sf.siftdown(start, end)
		if start == s {
			break
		}
		start--
	}
	end = e - 1
	for end > s {
		sf.Swap(s, end)
		sf.siftdown(s, end)
		end = end - 1
	}

}
