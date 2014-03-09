package CloudForest

import (
	"fmt"
	"log"
	"math"
	"strconv"
)

//Keeps track of votes by trees.
//Voteing is thread safe.
type NumBallotBox struct {
	box []*RunningMean
}

//Build a new ballot box for the number of cases specified by "size".
func NewNumBallotBox(size int) *NumBallotBox {
	bb := NumBallotBox{
		make([]*RunningMean, 0, size)}
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, new(RunningMean))
	}
	return &bb
}

//Vote parses the float in the string and votes for it
func (bb *NumBallotBox) Vote(casei int, pred string, weight float64) {
	v, err := strconv.ParseFloat(pred, 64)
	if err == nil {
		bb.box[casei].WeightedAdd(v, weight)
	}

}

//TallyNumerical tallies the votes for the case specified by i as
//if it is a Numerical feature. Ie it returns the mean of all votes.
func (bb *NumBallotBox) TallyNum(i int) (predicted float64) {
	predicted, _ = bb.box[i].Read()
	return
}

func (bb *NumBallotBox) Tally(i int) (predicted string) {
	mean, count := bb.box[i].Read()
	if count > 0 {
		predicted = fmt.Sprintf("%v", mean)
	} else {
		predicted = "NA"
	}
	return
}

//Tally error returns the error of the votes vs the provided feature.
//For categorical features it returns the error rate
//For numerical features it returns mean squared error.
//The provided feature must use the same index as the feature matrix
//the ballot box was constructed with.
//Missing values are ignored.
//Gini impurity is not used so this is not for use in rf implementations.
func (bb *NumBallotBox) TallyError(feature Feature) (e float64) {
	e = 0.0

	// Numerical feature. Calculate mean squared
	d := 0.0
	c := 0
	for i := 0; i < feature.Length(); i++ {
		predicted := bb.TallyNum(i)
		if !feature.IsMissing(i) && !math.IsNaN(predicted) {
			value := feature.(NumFeature).Get(i)

			d = float64(value) - predicted
			e += d * d
			c += 1
		}
	}
	if c == 0.0 {
		log.Fatal("TallyError with 0 count!")
	}
	e = e / float64(c)

	return

}

//Tally score returns the R2 score
func (bb *NumBallotBox) TallyR2Score(feature Feature) (e float64) {
	mean := 0.0
	r2 := 0.0
	total := 0
	for i := 0; i < feature.Length(); i++ {
		if !feature.IsMissing(i) {
			mean += feature.(*DenseNumFeature).Get(i)
			total++
		}
	}
	mean /= float64(total)

	for i := 0; i < feature.Length(); i++ {
		if !feature.IsMissing(i) {
			value := feature.(NumFeature).Get(i)

			d := float64(value) - mean
			r2 += d * d
		}
	}
	r2 /= float64(total)

	e = 1 - bb.TallyError(feature)/r2

	return

}
