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
		bb = CloudForest.NewNumBallotBox(len(data.Data[0].Missing))
	case false:
		bb = CloudForest.NewCatBallotBox(len(data.Data[0].Missing))

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}

	fmt.Printf("Outputting Predictions to %v\n", *predfn)
	for i, l := range data.CaseLabels {
		fmt.Fprintf(predfile, "%v\t%v\n", l, bb.Tally(i))
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		er := bb.TallyError(&data.Data[targeti])
		fmt.Printf("Error Rate: %v\n", er)
	}

}
