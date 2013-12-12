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
		"", "The name of a file to write the predictions into.")
	var num bool
	flag.BoolVar(&num, "mean", false, "Force numeric (mean) voting.")
	var cat bool
	flag.BoolVar(&cat, "mode", false, "Force categorical (mode) voting.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

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

	var predfile *os.File
	if *predfn != "" {
		predfile, err = os.Create(*predfn)
		if err != nil {
			log.Fatal(err)
		}
		defer predfile.Close()
	}

	var bb CloudForest.VoteTallyer
	if !cat && (num || strings.HasPrefix(forest.Target, "N")) {
		bb = CloudForest.NewNumBallotBox(data.Data[0].Length())
	} else {
		bb = CloudForest.NewCatBallotBox(data.Data[0].Length())

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		er := bb.TallyError(data.Data[targeti])
		fmt.Printf("%v\n", er)
	}
	if *predfn != "" {
		fmt.Printf("Outputting label predicted actual tsv o %v\n", *predfn)
		for i, l := range data.CaseLabels {
			actual := "NA"
			if hasTarget {
				actual = data.Data[targeti].GetStr(i)
			}
			fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, bb.Tally(i), actual)
		}
	}

}
