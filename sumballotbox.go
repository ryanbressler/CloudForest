package CloudForest

import (
	"fmt"
	"strconv"
	"sync"
)

//CatBallot is used insideof CatBallotBox to record catagorical votes in a thread safe
//manner.
type SumBallot struct {
	Mutex sync.Mutex
	Sum   float64
}

//NewCatBallot returns a pointer to an initalized CatBallot with a 0 size Map.
func NewSumBallot() (cb *SumBallot) {
	cb = new(SumBallot)
	cb.Sum = 0.0
	return
}

//CatBallotBox keeps track of votes by trees in a thread safe manner.
type SumBallotBox struct {
	Box []*SumBallot
}

//NewCatBallotBox builds a new ballot box for the number of cases specified by "size".
func NewSumBallotBox(size int) *SumBallotBox {
	bb := SumBallotBox{
		make([]*SumBallot, 0, size)}
	for i := 0; i < size; i++ {
		bb.Box = append(bb.Box, NewSumBallot())
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to be the
//category "pred".
func (bb *SumBallotBox) Vote(casei int, pred string, weight float64) {
	v, err := strconv.ParseFloat(pred, 64)
	if err == nil {

		bb.Box[casei].Mutex.Lock()
		bb.Box[casei].Sum += v * weight
		bb.Box[casei].Mutex.Unlock()
	}
}

//Tally tallies the votes for the case specified by i as
//if it is a Categorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *SumBallotBox) Tally(i int) (predicted string) {
	predicted = "NA"
	predicted = fmt.Sprintf("%v", bb.TallyNum(i))

	return

}

func (bb *SumBallotBox) TallyNum(i int) (predicted float64) {
	bb.Box[i].Mutex.Lock()
	predicted = bb.Box[i].Sum
	bb.Box[i].Mutex.Unlock()

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
func (bb *SumBallotBox) TallyError(feature Feature) (e float64) {

	return 1.0

}
