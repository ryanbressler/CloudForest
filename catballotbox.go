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

//VoteCat registers a vote that case "casei" should be predicted to have the
//catagorical "vote". 
func (bb *CatBallotBox) Vote(casei int, vote string) {
	voten := bb.CatToNum(vote)
	if _, ok := bb.box[casei][voten]; !ok {
		bb.box[casei][voten] = 0
	}
	bb.box[casei][voten] = bb.box[casei][voten] + 1
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

//Tally error returns the error of the votes vs the provided feature.
//For catagorical features it returns the error rate
//For numerical features it returns root mean squared error.
//The provided feature must use the same index as the feature matrix 
//the ballot box was constructed with.
//Missing values are ignored.
//Gini imurity is not used so this is not for use in rf implementations.
func (bb *CatBallotBox) TallyError(feature *Feature) (e float64) {
	e = 0.0

	//Catagorical feature. Calculate error rate
	c := 0
	for i, value := range feature.CatData {
		predicted := bb.Tally(i)
		if !feature.Missing[i] {
			if feature.Back[value] != predicted {
				e += 1.0
			}

			c += 1
		}
	}
	e = e / float64(c)

	return

}
