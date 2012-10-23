package CloudForest

import (
	"fmt"
	"io"
	"log"
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
