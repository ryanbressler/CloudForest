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
		"rface.sf", "A predictor forest as outputed by rf-ace")

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

	target := &data.Data[data.Map[forest.Target]]
	var bb CloudForest.VoteTallyer
	switch target.Numerical {
	case true:
		bb = CloudForest.NewNumBallotBox(len(data.Data[0].Missing))
	case false:
		bb = CloudForest.NewCatBallotBox(len(data.Data[0].Missing))

	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}
	er := bb.TallyError(target)
	fmt.Printf("%v\n", er)

}
