package CloudForest

import (
	"sync"
)

//CatBallot is used insideof CatBallotBox to record catagorical votes in a thread safe
//manner.
type CatBallot struct {
	Mutex sync.Mutex

	// If we use a map to tally votes, range picks categories in
	// a non-deterministic sequence. With a map, when ties occur in votes
	// between two categories, the first category evaluated will win.
	// Since, with a map, the
	// first category evaluated varies from run to run,
	// this causes sporadic test failures when
	// comparing the error rates on identical copies of trees.
	//
	// To obtain instead a deterministic voting process so
	// that we get deterministic tests,
	// we use a slice with a fixed category order.
	//
	// As a result, ties are presently awarded to the category
	// appearing earliest in Cat.
	//
	Cat []float64
}

//NewCatBallot returns a pointer to an initalized CatBallot with an empty (nil) Cat slice.
func NewCatBallot() (cb *CatBallot) {
	return &CatBallot{}
}

//CatBallotBox keeps track of votes by trees in a thread safe manner.
type CatBallotBox struct {
	*CatMap
	Box []*CatBallot
}

//NewCatBallotBox builds a new ballot box for the number of cases specified by "size".
func NewCatBallotBox(size int) *CatBallotBox {
	bb := CatBallotBox{
		&CatMap{make(map[string]int),
			make([]string, 0, 0)},
		make([]*CatBallot, 0, size)}
	for i := 0; i < size; i++ {
		bb.Box = append(bb.Box, NewCatBallot())
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to be the
//category "pred".
func (bb *CatBallotBox) Vote(casei int, pred string, weight float64) {

	predn := bb.CatToNum(pred)
	bb.Box[casei].Mutex.Lock()
	for len(bb.Box[casei].Cat) <= predn {
		bb.Box[casei].Cat = append(bb.Box[casei].Cat, 0)
	}
	bb.Box[casei].Cat[predn] += weight
	bb.Box[casei].Mutex.Unlock()
}

//Tally tallies the votes for the case specified by i as
//if it is a Categorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *CatBallotBox) Tally(i int) (predicted string) {
	predictedn := 0
	votes := 0.0
	bb.Box[i].Mutex.Lock()
	for k, v := range bb.Box[i].Cat {
		if v > votes {
			predictedn = k
			votes = v

		}

	}
	bb.Box[i].Mutex.Unlock()
	if votes > 0 {
		predicted = bb.Back[predictedn]
	} else {
		predicted = "NA"
	}

	return

}

/*
TallyError returns the balanced classification error for categorical features.

1 - sum((sum(Y(xi)=Y'(xi))/|xi|))

where
Y are the labels
Y' are the estimated labels
xi is the set of samples with the ith actual label

Case for which the true category is not known are ignored.

*/
func (bb *CatBallotBox) TallyError(feature Feature) (e float64) {
	catfeature := feature.(CatFeature)
	ncats := catfeature.NCats()
	correct := make([]int, ncats)
	total := make([]int, ncats)
	e = 0.0

	for i := 0; i < feature.Length(); i++ {
		value := catfeature.Geti(i)

		predicted := bb.Tally(i)
		if !feature.IsMissing(i) {
			total[value]++
			if catfeature.NumToCat(value) == predicted {
				correct[value]++
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
