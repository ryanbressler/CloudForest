package CloudForest

import (
	"io"
	"strings"
	"testing"
)

func TestFileFormats(t *testing.T) {

	//Write out a fm and read it back in
	pipereader, pipewriter := io.Pipe()
	cases := []int{0, 1, 2, 3, 4, 5, 6, 7}
	candidates := []int{2, 3, 4}

	fm1 := ParseAFM(strings.NewReader(fm))

	go func() {
		fm1.WriteCases(pipewriter, cases)
		pipewriter.Close()
	}()

	fm := ParseAFM(pipereader)

	if len(fm.Data) != 5 || fm.Data[0].Length() != 8 {
		t.Errorf("Iris feature matrix has %v features and %v cases not 5 and 8", len(fm.Data), fm.Data[0].Length())
	}

	cattarget := fm.Data[1]
	forest := GrowRandomForest(fm, cattarget.(Feature), candidates, fm.Data[0].Length(), 3, 10, 1, 0, false, false, false, false, nil)

	count := 0
	for _, tree := range forest.Trees {

		tree.Root.Recurse(func(*Node, []int, int) { count++ }, fm, cases, 0)

	}
	if count < 30 {
		t.Errorf("Trees before send to file has only %v nodes.", count)
	}

	pipereader, pipewriter = io.Pipe()

	go func() {
		fw := NewForestWriter(pipewriter)
		fw.WriteForest(forest)
		pipewriter.Close()
	}()

	fr := NewForestReader(pipereader)

	forest, err := fr.ReadForest()
	if err != nil {
		t.Errorf("Error parseing forest from pipe: %v", err)
	}
	if len(forest.Trees) != 10 {
		t.Errorf("Parsed forrest has only %v trees.", len(forest.Trees))
	}

	catvotes := NewCatBallotBox(cattarget.Length())
	count2 := 0
	for _, tree := range forest.Trees {
		tree.Vote(fm, catvotes)
		tree.Root.Recurse(func(*Node, []int, int) { count2++ }, fm, cases, 0)

	}
	if count != count2 {
		t.Errorf("Forest before file has %v nodes differs form %v nodes after.", count, count2)
	}

	//TODO(ryan): figure out what is going on with go 1.3 and use more stringent threshold here
	score := catvotes.TallyError(cattarget)
	if score > 0.4 {
		t.Errorf("Error: Classification of simpledataset from sf file had score: %v", score)
	}

}
