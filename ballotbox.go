package CloudForest

import (
	"math"
)

//A structur for keeping track of votes by trees.
//Not thread safe....could be made so or abstracted to an 
//interface to support diffrent implementations.
type BallotBox struct {
	box []map[Num]int
}

//Build a new ballot box of the specified size.
func NewBallotBox(size int) *BallotBox {
	bb := new(BallotBox)
	bb.box = make([]map[Num]int, 0, size)
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, make(map[Num]int, 0))
	}
	return bb
}

//Vote increments the vote in the ballot box for the provided case
//and predictor value "vote." The predictor value "vote" must be provided in
//the Numerical value used in the feature matrix being voted on.
func (bb *BallotBox) Vote(casei int, vote Num) {

	if _, ok := bb.box[casei][vote]; !ok {
		bb.box[casei][vote] = 0
	}
	bb.box[casei][vote] = bb.box[casei][vote] + 1
}

//TallyNumerical tallies the votes for the case specified by i as
//if it is a Numerical feature. Ie it returns the weighted mean of all votes.
func (bb *BallotBox) TallyNumerical(i int) (predicted float64) {
	predicted = 0.0
	votes := 0
	for k, v := range bb.box[i] {
		predicted += float64(k) * float64(v)
		votes += v

	}
	predicted = predicted / float64(votes)
	return
}

//TallyCatagorical tallies the votes for the case specified by i as 
//if it is a Catagorical or boolean feature. Ie it returns the weighted
//mode of all votes.
func (bb *BallotBox) TallyCatagorical(i int) (predicted float64) {
	votes := 0
	for k, v := range bb.box[i] {
		if v > votes {
			predicted = float64(k)

		}

	}
	return

}

//Tally error returns the error of the votes vs the provided feature.
//For catagorical features it returns the error rate
//For numerical features it returns root mean squared error.
//The provided feature must use the same index as the feature matrix 
//the ballot box was constructed with.
//Missing values are ignored.
func (bb *BallotBox) TallyError(feature *Feature) (e float64) {
	e = 0.0
	switch feature.Numerical {
	case true:
		// Numerical feature. Calculate root mean squared
		d := 0.0
		c := 0
		for i, value := range feature.Data {
			if !feature.Missing[i] {
				d = float64(value) - bb.TallyNumerical(i)
				e += d * d
				c += 1
			}
		}
		e = math.Sqrt(e / float64(c))
	case false:
		//Catagorical feature. Calculate error rate
		c := 0
		for i, value := range feature.Data {
			if !feature.Missing[i] {
				if int(value) != int(bb.TallyCatagorical(i)) {
					e += 1.0
				}

				c += 1
			}
		}
		e = e / float64(c)

	}
	return

}
