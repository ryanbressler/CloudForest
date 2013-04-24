package CloudForest

import ()

/*
Sortable feature is a wrapper for a feature and set of cases that satisfies the
sort.Interface interface so that the case indexes in Cases can be sorted using
sort.Sort
*/
type SortableFeature struct {
	feature *Feature
	Cases   []int
}

//Len returns the number of cases.
func (sf SortableFeature) Len() int {
	return len(sf.Cases)
}

//Less determines if the ith case is less then the jth case.
func (sf SortableFeature) Less(i int, j int) bool {
	return sf.feature.NumData[sf.Cases[i]] < sf.feature.NumData[sf.Cases[j]]

}

//Swap exchanges the ith and jth cases.
func (sf SortableFeature) Swap(i int, j int) {
	v := sf.Cases[i]
	sf.Cases[i] = sf.Cases[j]
	sf.Cases[j] = v

}
