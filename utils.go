package CloudForest

import (
	"fmt"
	"io"
	"log"
	"math/rand"
)

type SparseCounter struct {
	Map map[int]map[int]int
}

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
SampleFirstN insures that the first n entries in the supplied 
"deck" []int are randomly drawn from all entries without replacment. 
It accepts a pointer to the deck so that it can be used repeatedl on
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
