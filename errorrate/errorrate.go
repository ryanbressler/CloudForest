package main

import (
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing test data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest.")

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

	target := data.Data[data.Map[forest.Target]]
	var bb CloudForest.VoteTallyer
	if target.NCats() == 0 {
		bb = CloudForest.NewNumBallotBox(data.Data[0].Length())
	} else {
		bb = CloudForest.NewCatBallotBox(data.Data[0].Length())

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}
	er := bb.TallyError(target)
	fmt.Printf("%v\n", er)

}
