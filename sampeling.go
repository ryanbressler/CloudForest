package CloudForest

import (
	"math/rand"
)

type BalancedSampler struct {
	Cases [][]int
}

func NewBalancedSampler(catf *DenseCatFeature) (s *BalancedSampler) {
	s = &BalancedSampler{make([][]int, catf.NCats())}

	for i := 0; i < catf.NCats(); i++ {
		s.Cases = append(s.Cases, make([]int, catf.Length()))
	}

	for i, v := range catf.CatData {
		if !catf.IsMissing(i) {
			s.Cases[v] = append(s.Cases[v], i)
		}
	}
	return
}

func (s *BalancedSampler) Sample(samples *[]int, n int) {
	(*samples) = (*samples)[0:0]
	nCases := len(s.Cases)
	c := 0
	for i := 0; i < n; i++ {
		c = rand.Intn(nCases)
		(*samples) = append((*samples), s.Cases[c][rand.Intn(len(s.Cases[c]))])
	}

}

/*
SampleFirstN ensures that the first n entries in the supplied
deck are randomly drawn from all entries without replacement for use in selecting candidate
features to split on. It accepts a pointer to the deck so that it can be used repeatedly on
the same deck avoiding reallocations.
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
SampleWithReplacment samples nSamples random draws from [0,totalCases) with replacement
for use in selecting cases to grow a tree from.
*/
func SampleWithReplacment(nSamples int, totalCases int) (cases []int) {
	cases = make([]int, 0, nSamples)
	for i := 0; i < nSamples; i++ {
		cases = append(cases, rand.Intn(totalCases))
	}
	return
}
