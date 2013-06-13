package main

import (
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest.")
	predfn := flag.String("preds",
		"predictions.tsv", "The name of a file to write the predictions into.")

	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := CloudForest.ParseAFM(datafile)

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestreader := CloudForest.NewForestReader(forestfile)
	forest, err := forestreader.ReadForest()
	if err != nil {
		log.Fatal(err)
	}

	predfile, err := os.Create(*predfn)
	if err != nil {
		log.Fatal(err)
	}

	var bb CloudForest.VoteTallyer
	switch strings.HasPrefix(forest.Target, "N") {
	case true:
		bb = CloudForest.NewNumBallotBox(data.Data[0].Length())
	case false:
		bb = CloudForest.NewCatBallotBox(data.Data[0].Length())

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		er := bb.TallyError(data.Data[targeti])
		fmt.Printf("Error Rate: %v\n", er)
	}

	fmt.Printf("Outputting label\tpredicted\tactual to %v\n", *predfn)
	for i, l := range data.CaseLabels {
		actual := "NA"
		if hasTarget {
			actual = data.Data[targeti].GetStr(i)
		}
		fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, bb.Tally(i), actual)
	}

}
