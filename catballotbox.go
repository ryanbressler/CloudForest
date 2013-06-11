package CloudForest

import (
	"sync"
)

//Cat ballot is used insideof CatBallotBox to record catagorical votes in a thread safe
//manner.
type CatBallot struct {
	Mutex sync.Mutex
	Map   map[int]float64
}

//NewCatBallot returns a pointer to an initalized CatBallot with a 0 size Map.
func NewCatBallot() (cb *CatBallot) {
	cb = new(CatBallot)
	cb.Map = make(map[int]float64, 0)
	return
}

//Keeps track of votes by trees in a thread safe manner.
type CatBallotBox struct {
	*CatMap
	box []*CatBallot
}

//Build a new ballot box for the number of cases specified by "size".
func NewCatBallotBox(size int) *CatBallotBox {
	bb := CatBallotBox{
		&CatMap{make(map[string]int),
			make([]string, 0, 0)},
		make([]*CatBallot, 0, size)}
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, NewCatBallot())
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to be the
//category "pred".
func (bb *CatBallotBox) Vote(casei int, pred string, weight float64) {
	predn := bb.CatToNum(pred)
	bb.box[casei].Mutex.Lock()
	if _, ok := bb.box[casei].Map[predn]; !ok {
		bb.box[casei].Map[predn] = 0
	}
	bb.box[casei].Map[predn] = bb.box[casei].Map[predn] + weight
	bb.box[casei].Mutex.Unlock()
}

//TallyCatagorical tallies the votes for the case specified by i as
//if it is a Categorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *CatBallotBox) Tally(i int) (predicted string) {
	predictedn := 0
	votes := 0.0
	bb.box[i].Mutex.Lock()
	for k, v := range bb.box[i].Map {
		if v > votes {
			predictedn = k
			votes = v

		}

	}
	bb.box[i].Mutex.Unlock()
	predicted = bb.Back[predictedn]
	return

}

/*
Tally error returns the balanced classification error for categorical features.

1 - sum((sum(Y(xi)=Y'(xi))/|xi|))

where
Y are the labels
Y' are the estimated labels
xi is the set of samples with the ith actual label

Case for which the true category is not known are ignored.

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
