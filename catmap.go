package CloudForest

import ()

/*CatMap is for mapping categorical values to integers.
It contains:

	Map  : a map of ints by the string used for the category
	Back : a slice of strings by the int that represents them

And is embedded by Feature and CatBallotBox.
*/
type CatMap struct {
	Map  map[string]int //map categories from string to Num
	Back []string       // map categories from Num to string
}

//CatToNum provides the int equivalent of the provided categorical value
//if it already exists or adds it to the map and returns the new value if
//it doesn't.
func (cm *CatMap) CatToNum(value string) (numericv int) {
	numericv, exsists := cm.Map[value]
	if exsists == false {
		numericv = len(cm.Back)
		cm.Map[value] = numericv
		cm.Back = append(cm.Back, value)

	}
	return
}

func (cm *CatMap) NCats() (n int) {
	if cm.Back == nil {
		n = 0
	} else {
		n = len(cm.Back)
	}
	return
}
