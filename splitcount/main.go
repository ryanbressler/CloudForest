package main

import (
	"flag"
	"github.com/rbkreisberg/CloudForest"
	"log"
	"os"
	"fmt"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest as outputed by rf-ace")
	outf := flag.String("splits", "splits.tsv", "a case by case sparse matrix of leaf cooccurance in tsv format")
	boutf := flag.String("branches", "branches.tsv", "a case by feature sparse matrix of case/splitter cooccurance in tsv format")
	rboutf := flag.String("branches", "relativeBranches.tsv", "a case by feature sparse matrix of split direction for each case/feature in tsv format")

	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := CloudForest.ParseAFM(datafile)
	nfeatures := len(data.Data)
	ncases := len(data.Data[0].Data)
	log.Print("Data file ", nfeatures, " by ", ncases)

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forest := CloudForest.ParseRfAcePredictor(forestfile)
	log.Print("Forest has ", len(forest.Trees), " trees ")

	featureCounts := make([]int, nfeatures) // a total count of the number of times each feature was used to split
	caseFeatureCounts := new(CloudForest.SparseCounter)
	relativeSplitCount := new(CloudForest.SparseCounter)

	for i := 0; i < len(forest.Trees); i++ {
		splits := forest.Trees[i].GetSplits(data, caseFeatureCounts, relativeSplitCount)

		for _, split := range splits {
			featureCounts[data.Map[split.Feature]]++ //increment the count for the total # of times the feature was a splitter
		}
	}

	log.Print("Outputing Split Feature/Case Cooccurance Counts")
	outfile, err := os.Create(*outf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()
	for feature, count := range featureCounts {
		if _, err := fmt.Fprintf(outfile, "%v\t%v\n", feature, count); err != nil {
			log.Fatal(err)
		}
	}

	log.Print("Outputing Case Feature Cooccurance Counts")
	boutfile, err := os.Create(*boutf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer boutfile.Close()
	caseFeatureCounts.WriteTsv(boutfile)

	log.Print("Outputing Case Feature Splitter Direction")
	rboutfile, err := os.Create(*rboutf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer rboutfile.Close()
	relativeSplitCount.WriteTsv(boutfile)
}
