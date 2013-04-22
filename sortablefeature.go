package CloudForest

import ()

type SortableFeature struct {
	feature *Feature
	Cases   []int
}

func (sf SortableFeature) Len() int {
	return len(sf.Cases)
}

func (sf SortableFeature) Less(i int, j int) bool {
	return sf.feature.NumData[i] < sf.feature.NumData[j]

}

func (sf SortableFeature) Swap(i int, j int) {
	v := sf.Cases[i]
	sf.Cases[i] = j
	sf.Cases[j] = v

}
