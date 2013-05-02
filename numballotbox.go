package CloudForest

import (
	"fmt"
	"math"
	"strconv"
)

//Keeps track of votes by trees.
//Not thread safe....could be made so or abstracted to an
//interface to support diffrent implementations.
type NumBallotBox struct {
	box []map[float64]int
}

//Build a new ballot box for the number of cases specified by "size".
func NewNumBallotBox(size int) *NumBallotBox {
	bb := NumBallotBox{
		make([]map[float64]int, 0, size)}
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, make(map[float64]int, 0))
	}
	return &bb
}

//Vote parses the float in the string and votes for it
func (bb *NumBallotBox) Vote(casei int, pred string) {
	v, err := strconv.ParseFloat(pred, 64)
	if err == nil {
		bb.VoteNum(casei, v)
	}

}

//VoteNum registers a vote that case "casei" should be predicted to have the
//numerical value "vote."
func (bb *NumBallotBox) VoteNum(casei int, pred float64) {

	if _, ok := bb.box[casei][pred]; !ok {
		bb.box[casei][pred] = 0
	}
	bb.box[casei][pred] = bb.box[casei][pred] + 1
}

//TallyNumerical tallies the votes for the case specified by i as
//if it is a Numerical feature. Ie it returns the mean of all votes.
func (bb *NumBallotBox) TallyNum(i int) (predicted float64) {
	predicted = 0.0
	votes := 0
	for k, v := range bb.box[i] {
		predicted += float64(k) * float64(v)
		votes += v

	}
	predicted = predicted / float64(votes)
	return
}

func (bb *NumBallotBox) Tally(i int) (predicted string) {
	return fmt.Sprintf("%v", bb.TallyNum(i))
}

//Tally error returns the error of the votes vs the provided feature.
//For catagorical features it returns the error rate
//For numerical features it returns mean squared error.
//The provided feature must use the same index as the feature matrix
//the ballot box was constructed with.
//Missing values are ignored.
//Gini imurity is not used so this is not for use in rf implementations.
func (bb *NumBallotBox) TallyError(feature *Feature) (e float64) {
	e = 0.0

	// Numerical feature. Calculate mean squared
	d := 0.0
	c := 0
	for i, value := range feature.NumData {
		predicted := bb.TallyNum(i)
		if !feature.Missing[i] && !math.IsNaN(predicted) {
			d = float64(value) - predicted
			e += d * d
			c += 1
		}
	}
	e = e / float64(c)

	return

}
