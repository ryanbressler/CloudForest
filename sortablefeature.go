package CloudForest

import (
	//"fmt"
	//"math"
	"github.com/lytics/CloudForest/sortby"
	//"sort"
)

/*SortableFeature is a wrapper for a feature and set of cases that satisfies the
sort.Interface interface so that the case indexes in Cases can be sorted using
sort.Sort
*/
type SortableFeature struct {
	//Feature NumFeature
	Vals  []float64
	Cases []int
}

//Sort performs introsort + heapsort using the sortby sub package.
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

//Load loads the values of the cases into a cache friendly array.
func (sf *SortableFeature) Load(vals *[]float64, cases *[]int) {
	sf.Cases = *cases
	sfvals := sf.Vals
	vs := *vals
	for i, p := range *cases {
		sfvals[i] = vs[p]
	}
}
