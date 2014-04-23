package CloudForest

import (
	"strings"
	"testing"
)

func TestSampleFirstN(t *testing.T) {
	deck := []int{0, 1, 2, 3}
	var samples []int
	SampleFirstN(&deck, &samples, 2, 0)
	if len(samples) != 2 {
		t.Errorf("Error: sampeling 2 items returned %v samples", len(samples))
	}
	deck = []int{0, 1, 2, 3}
	SampleFirstN(&deck, &samples, 2, 2)

	if deck[2] != 2 || deck[3] != 3 {
		t.Errorf("Sampeling 2 items with 2 constant resulted in %v %v", deck, samples)
	}

	deck = []int{0, 1, 2, 3}
	SampleFirstN(&deck, &samples, 2, 3)

	if deck[1] != 1 || deck[2] != 2 || deck[3] != 3 {
		t.Errorf("Sampeling 2 items with 3 constant resulted in %v %v", deck, samples)
	}

}

var bfm = `.	0	1	2	3	4	5	6	7
C:1	0	0	1	1	1	1	1	1
C:2	0	1	0	1	0	1	0	1`

func TestSampeling(t *testing.T) {
	fmReader := strings.NewReader(bfm)

	fm := ParseAFM(fmReader)
	cases := make([]int, 0, 1000)

	samplers := []Bagger{NewBalancedSampler(fm.Data[0].(*DenseCatFeature)),
		NewSecondaryBalancedSampler(fm.Data[0].(*DenseCatFeature), fm.Data[1].(*DenseCatFeature)),
	}

	for _, bs := range samplers {
		bs.Sample(&cases, 1000)
		case0 := 0
		case1 := 0

		for _, c := range cases {
			if c == 0 {
				case0++
			}
			if c == 1 {
				case1++
			}
		}
		switch bs.(type) {
		case *BalancedSampler:
			s := bs.(*BalancedSampler)
			if l := len(s.Cases); l != 2 {
				t.Errorf("Balanced sampler found %v cases not 2: %v", l, fm.Data[0].(*DenseCatFeature).Back)
			}

		case *SecondaryBalancedSampler:
			s := bs.(*SecondaryBalancedSampler)
			if s.Total != 8 {
				t.Errorf("SecondaryBalanced sampler found %v total cases not 8", s.Total)
			}
			if l := len(s.Samplers); l != 2 {
				t.Errorf("SecondaryBalanced sampler found %v cases not 2", l)
			}
			if l := len(s.Counts); l != 2 {
				t.Errorf("SecondaryBalanced sampler found %v cases not 2", l)
			}
		}
		if case0 < 200 || case1 < 200 {
			t.Errorf("Cases 0 and 1 underprepresented after balanced sampeling from %T.", bs)
		}

	}

}
