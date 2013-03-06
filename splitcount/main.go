package main

import (
	"flag"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest as outputed by rf-ace")
	outf := flag.String("splits", "splits.tsv", "a case by case sparse matrix of leaf cooccurance in tsv format")

	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := CloudForest.ParseAFM(datafile)
	log.Print("Data file ", len(data.Data), " by ", len(data.Data[0].Data))

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forest := CloudForest.ParseRfAcePredictor(forestfile)
	log.Print("Forest has ", len(forest.Trees), " trees ")

	counts := new(CloudForest.SparseCounter)
	caseFeatureCounts := new(CloudForest.SparseCounter)

	for i := 0; i < len(forest.Trees); i++ {
		leaves := forest.Trees[i].GetLeaves(data, caseFeatureCounts)
		for _, leaf := range leaves {
			for j := 0; j < len(leaf.Cases); j++ {
				for k := 0; k < len(leaf.Cases); k++ {

					counts.Add(leaf.Cases[j], leaf.Cases[k], 1)

				}
			}
		}

	}

	log.Print("Outputing Case Case  Cocurance Counts")
	outfile, err := os.Create(*outf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()
	counts.WriteTsv(outfile)

	log.Print("Outputing Case Feature Cocurance Counts")
	boutfile, err := os.Create(*boutf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer boutfile.Close()
	caseFeatureCounts.WriteTsv(boutfile)
}
