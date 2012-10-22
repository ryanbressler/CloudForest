package main

import (
	"flag"
	"log"
	"os"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest as outputed by rf-ace")
	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := ParseAFM(datafile)
	log.Print("Data file ", len(data), " by ", len(data[0].Data))

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forest := ParseRfAcePredictor(forestfile)
	log.Print("Fores has ", len(forest.Trees), " trees ")

}
