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
	forest := CloudForest.ParseRfAcePredictor(forestfile)

	bb := CloudForest.NewBallotBox(len(data.Data[0].Data))

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}
	er := bb.TallyError(&data.Data[data.Map[forest.Target]])
	fmt.Printf("%v\n", er)

}
