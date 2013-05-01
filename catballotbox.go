package CloudForest

import ()

//Keeps track of votes by trees.
//Not thread safe....could be made so or abstracted to an
//interface to support diffrent implementations.
type CatBallotBox struct {
	*CatMap
	box []map[int]int
}

//Build a new ballot box for the number of cases specified by "size".
func NewCatBallotBox(size int) *CatBallotBox {
	bb := CatBallotBox{
		&CatMap{make(map[string]int),
			make([]string, 0, 0)},
		make([]map[int]int, 0, size)}
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, make(map[int]int, 0))
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to be the
//catagory "pred".
func (bb *CatBallotBox) Vote(casei int, pred string) {
	predn := bb.CatToNum(pred)
	if _, ok := bb.box[casei][predn]; !ok {
		bb.box[casei][predn] = 0
	}
	bb.box[casei][predn] = bb.box[casei][predn] + 1
}

//TallyCatagorical tallies the votes for the case specified by i as
//if it is a Catagorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *CatBallotBox) Tally(i int) (predicted string) {
	predictedn := 0
	votes := 0
	for k, v := range bb.box[i] {
		if v > votes {
			predictedn = k
			votes = v

		}

	}
	predicted = bb.Back[predictedn]
	return

}

/*
Tally error returns the balanced clasification error for catagorical features.

1 - sum((sum(Y(xi)=Y'(xi))/|xi|))

where
Y are the labels
Y' are the estimated labels
xi is the set of samples with the ith actual label

Case for which the true catagory is not known are ignored.

*/
func (bb *CatBallotBox) TallyError(feature *Feature) (e float64) {
	ncats := feature.NCats()
	correct := make([]int, ncats)
	total := make([]int, ncats)
	e = 0.0

	for i, value := range feature.CatData {

		predicted := bb.Tally(i)
		if !feature.Missing[i] {
			total[value] += 1
			if feature.Back[value] == predicted {
				correct[value] += 1
			}

		}
	}

	for i, ncorrect := range correct {
		e += float64(ncorrect) / float64(total[i])
	}

	e /= float64(ncats)
	e = 1.0 - e

	return

}
