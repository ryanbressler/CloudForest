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
	sfvals := sf.Vals
	vs := *vals
	for i, p := range *cases {
		sfvals[i] = vs[p]
	}
}
