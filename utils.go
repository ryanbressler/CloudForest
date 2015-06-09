package CloudForest

import (
	"fmt"
	"io"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"
)

func ParseFloat(s string) float64 {
	frac, _ := strconv.ParseFloat(s, 64)
	return frac

}

//RunningMean is a thread safe strut for keeping track of running means as used in
//importance calculations. (TODO: could this be made lock free?)
type RunningMean struct {
	mutex sync.Mutex
	Mean  float64
	Count float64
}

//Add add's 1.0 to the running mean in a thread safe way.
func (rm *RunningMean) Add(val float64) {
	rm.WeightedAdd(val, 1.0)
}

//WeightedAdd add's the specified value to the running mean in a thread safe way.
func (rm *RunningMean) WeightedAdd(val float64, weight float64) {
	if !math.IsNaN(val) && !math.IsNaN(weight) {
		rm.mutex.Lock()
		rm.Mean = (rm.Mean*rm.Count + weight*val) / (rm.Count + weight)
		rm.Count += weight
		if rm.Count == 0 {
			log.Print("WeightedAdd reached 0 count!.")
		}
		if math.IsNaN(rm.Mean) || math.IsNaN(rm.Count) {
			log.Print("Weighted add reached nan after adding ", val, weight)
		}

		rm.mutex.Unlock()
	}

}

//Read reads the mean and count
func (rm *RunningMean) Read() (mean float64, count float64) {
	rm.mutex.Lock()
	mean = rm.Mean
	count = rm.Count
	rm.mutex.Unlock()
	return
}

//NewRunningMeans returns an initalized *[]*RunningMean.
func NewRunningMeans(size int) *[]*RunningMean {
	importance := make([]*RunningMean, 0, size)
	for i := 0; i < size; i++ {
		rm := new(RunningMean)
		importance = append(importance, rm)
	}
	return &importance

}

//SparseCounter uses maps to track sparse integer counts in large matrix.
//The matrix is assumed to contain zero values where nothing has been added.
type SparseCounter struct {
	Map   map[int]map[int]int
	mutex sync.Mutex
}

//Add increases the count in i,j by val.
func (sc *SparseCounter) Add(i int, j int, val int) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
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

//WriteTsv writes the non zero counts out into a three column tsv containing i, j, and
//count in the columns.
func (sc *SparseCounter) WriteTsv(writer io.Writer) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
	for i := range sc.Map {
		for j, val := range sc.Map[i] {
			if _, err := fmt.Fprintf(writer, "%v\t%v\t%v\n", i, j, val); err != nil {
				log.Fatal(err)
			}
		}
	}

}

/*
ParseAsIntOrFractionOfTotal parses strings that may specify an count or a percent of
the total for use in specifying paramaters.
It parses term as a float if it contains a "." and as an int otherwise. If term is parsed
as a float frac it returns int(math.Ceil(frac * float64(total))).
It returns zero if term == "" or if a parsing error occures.
*/
func ParseAsIntOrFractionOfTotal(term string, total int) (parsed int) {
	if term == "" {
		return 0
	}

	if strings.Contains(term, ".") {
		frac, err := strconv.ParseFloat(term, 64)
		if err == nil {
			parsed = int(math.Ceil(frac * float64(total)))
		} else {
			parsed = 0
		}
	} else {
		count, err := strconv.ParseInt(term, 0, 0)
		if err != nil {
			parsed = 0
		} else {
			parsed = int(count)
		}

	}
	return
}
