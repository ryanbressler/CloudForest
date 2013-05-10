package CloudForest

import (
	"fmt"
	"io"
	"log"
	"math/rand"
	"sync"
)

//RunningMean is a thread safe struct for keeping track of running means as used in
//importance calculations. (TODO: could this be made lock free?)
type RunningMean struct {
	*sync.Mutex
	Mean  float64
	Count int
}

//RunningMean.Add add's the specifed value to the running mean in a thread safe way.
func (rm *RunningMean) Add(val float64) {
	rm.Lock()
	rm.Mean = (rm.Mean*float64(rm.Count) + val) / (float64(rm.Count) + 1.0)
	rm.Count += 1
	rm.Unlock()
}

//Sparse counter uses maps to track sparse integer counts in large matrix.
//The matrix is assumed to contain zero values where nothing has been edded.
type SparseCounter struct {
	Map map[int]map[int]int
}

//Add increases the count in i,j by val.
func (sc *SparseCounter) Add(i int, j int, val int) {
	if sc.Map == nil {
		sc.Map = make(map[int]map[int]int, 0)
	}

	if v, ok := sc.Map[i]; !ok || v == nil {
		sc.Map[i] = make(map[int]int, 0)
	}
	if _, ok := sc.Map[i][j]; !ok {
		sc.Map[i][j] = 0
	}
	sc.Map[i][j] = sc.Map[i][j] + val
}

//Write tsv writes the non zero counts out into a three colum tsv containg i, j, and
//count in the columns.
func (sc *SparseCounter) WriteTsv(writer io.Writer) {
	for i := range sc.Map {
		for j, val := range sc.Map[i] {
			if _, err := fmt.Fprintf(writer, "%v\t%v\t%v\n", i, j, val); err != nil {
				log.Fatal(err)
			}
		}
	}
}

/*
SampleFirstN ensures that the first n entries in the supplied
deck are randomly drawn from all entries without replacment for use in selecting canidate
features to split on. It accepts a pointer to the deck so that it can be used repeatedl on
the same deck avoiding realocations.
*/
func SampleFirstN(deck *[]int, n int) {
	cards := *deck
	length := len(cards)
	old := 0
	randi := 0
	for i := 0; i < n; i++ {
		old = cards[i]
		randi = i + rand.Intn(length-i)
		cards[i] = cards[randi]
		cards[randi] = old

	}
}

/*
SampleWithReplacment samples nSamples random draws from [0,totalCases) with replacment
for use in selecting cases to grow a tree from.
*/
func SampleWithReplacment(nSamples int, totalCases int) (cases []int) {
	cases = make([]int, 0, nSamples)
	for i := 0; i < nSamples; i++ {
		cases = append(cases, rand.Intn(totalCases))
	}
	return
}
