package CloudForest

import ()

type SortableFeature struct {
	feature *Feature
	cases   []int
}

func (sf *SortableFeature) Len() int {
	return len(sf.cases)
}

func (sf *SortableFeature) Less(i int, j int) bool {
	return sf.feature.NumData[i] < sf.feature.NumData[j]

}

func (sf *SortableFeature) Swap(i int, j int) {

}
