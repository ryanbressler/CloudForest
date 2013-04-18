package CloudForest

import (
	"math"
)

//Keeps track of votes by trees.
//Not thread safe....could be made so or abstracted to an 
//interface to support diffrent implementations.
type BallotBox struct {
	*CatMap
	box []map[Num]int
}

//Build a new ballot box for the number of cases specified by "size".
func NewBallotBox(size int) *BallotBox {
	bb := BallotBox{
		&CatMap{make(map[string]Num),
			make(map[Num]string),
			0.0},
		make([]map[Num]int, 0, size)}
	for i := 0; i < size; i++ {
		bb.box = append(bb.box, make(map[Num]int, 0))
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to have the value
//"vote". 
func (bb *BallotBox) VoteNum(casei int, vote Num) {

	if _, ok := bb.box[casei][vote]; !ok {
		bb.box[casei][vote] = 0
	}
	bb.box[casei][vote] = bb.box[casei][vote] + 1
}

//Vote registers a vote that case "casei" should be predicted to have the value
//"vote". 
func (bb *BallotBox) VoteCat(casei int, vote string) {
	voten := bb.CatToNum(vote)
	if _, ok := bb.box[casei][voten]; !ok {
		bb.box[casei][voten] = 0
	}
	bb.box[casei][voten] = bb.box[casei][voten] + 1
}

//TallyNumerical tallies the votes for the case specified by i as
//if it is a Numerical feature. Ie it returns the mean of all votes.
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
//if it is a Catagorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *BallotBox) TallyCatagorical(i int) (predicted string) {
	predictedn := Num(0.0)
	votes := 0
	for k, v := range bb.box[i] {
		if v > votes {
			predictedn = k

		}

	}
	predicted = bb.Back[predictedn]
	return

}

//Tally error returns the error of the votes vs the provided feature.
//For catagorical features it returns the error rate
//For numerical features it returns root mean squared error.
//The provided feature must use the same index as the feature matrix 
//the ballot box was constructed with.
//Missing values are ignored.
//Gini imurity is not used so this is not for use in rf implementations.
func (bb *BallotBox) TallyError(feature *Feature) (e float64) {
	e = 0.0
	switch feature.Numerical {
	case true:
		// Numerical feature. Calculate root mean squared
		d := 0.0
		c := 0
		for i, value := range feature.Data {
			predicted := bb.TallyNumerical(i)
			if !feature.Missing[i] && !math.IsNaN(predicted) {
				d = float64(value) - predicted
				e += d * d
				c += 1
			}
		}
		e = math.Sqrt(e / float64(c))
	case false:
		//Catagorical feature. Calculate error rate
		c := 0
		for i, value := range feature.Data {
			predicted := bb.TallyCatagorical(i)
			if !feature.Missing[i] {
				if feature.Back[value] != predicted {
					e += 1.0
				}

				c += 1
			}
		}
		e = e / float64(c)

	}
	return

}
