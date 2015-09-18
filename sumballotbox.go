package CloudForest

import (
	"fmt"
	"strconv"
	"sync"
)

//SumBallot is used insideof SumBallotBox to record sum votes in a thread safe
//manner.
type SumBallot struct {
	Mutex sync.Mutex
	Sum   float64
}

//NewSumBallot returns a pointer to an initalized SumBallot with a 0 size Map.
func NewSumBallot() (cb *SumBallot) {
	cb = new(SumBallot)
	cb.Sum = 0.0
	return
}

//SumBallotBox keeps track of votes by trees in a thread safe manner.
//It should be used with gradient boosting when a sum instead of an average
//or mode is desired.
type SumBallotBox struct {
	Box []*SumBallot
}

//NewSumBallotBox builds a new ballot box for the number of cases specified by "size".
func NewSumBallotBox(size int) *SumBallotBox {
	bb := SumBallotBox{
		make([]*SumBallot, 0, size)}
	for i := 0; i < size; i++ {
		bb.Box = append(bb.Box, NewSumBallot())
	}
	return &bb
}

//Vote registers a vote that case "casei" should have pred added to its
//sum.
func (bb *SumBallotBox) Vote(casei int, pred string, weight float64) {
	v, err := strconv.ParseFloat(pred, 64)
	if err == nil {

		bb.Box[casei].Mutex.Lock()
		bb.Box[casei].Sum += v * weight
		bb.Box[casei].Mutex.Unlock()
	}
}

//Tally tallies the votes for the case specified by i as
//if it is a Categorical or boolean feature. Ie it returns the sum
//of all votes.
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
TallyError is non functional here.
*/
func (bb *SumBallotBox) TallyError(feature Feature) (e float64) {

	return 1.0

}
