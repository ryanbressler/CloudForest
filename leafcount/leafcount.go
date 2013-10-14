package main

import (
	"flag"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest.")
	outf := flag.String("leaves", "leaves.tsv", "a case by case sparse matrix of leaf co-occurrence in tsv format")
	boutf := flag.String("branches", "branches.tsv", "a case by feature sparse matrix of leaf co-occurrence in tsv format")

	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := CloudForest.ParseAFM(datafile)
	log.Print("Data file ", len(data.Data), " by ", data.Data[0].Length())

	counts := new(CloudForest.SparseCounter)
	caseFeatureCounts := new(CloudForest.SparseCounter)

	for _, fn := range strings.Split(*rf, ",") {

		forestfile, err := os.Open(fn) // For read access.
		if err != nil {
			log.Fatal(err)
		}
		defer forestfile.Close()
		forestreader := CloudForest.NewForestReader(forestfile)
		forest, err := forestreader.ReadForest()
		if err != nil {
			log.Fatal(err)
		}
		log.Print("Forest has ", len(forest.Trees), " trees ")

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
	}

	log.Print("Outputting Case Case  Co-Occurrence Counts")
	outfile, err := os.Create(*outf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()
	counts.WriteTsv(outfile)

	log.Print("Outputting Case Feature Co-Occurrence Counts")
	boutfile, err := os.Create(*boutf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer boutfile.Close()
	caseFeatureCounts.WriteTsv(boutfile)
}
