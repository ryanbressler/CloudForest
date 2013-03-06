package main

import (
	"flag"
	"fmt"
	"github.com/rbkreisberg/CloudForest"
	"log"
	"os"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest as outputed by rf-ace")
	outf := flag.String("splits", "splits.tsv", "a case by case sparse matrix of leaf cooccurance in tsv format")
	boutf := flag.String("branches", "branches.tsv", "a case by feature sparse matrix of case/splitter cooccurance in tsv format")
	rboutf := flag.String("relbranches", "relativeBranches.tsv", "a case by feature sparse matrix of split direction for each case/feature in tsv format")
	splitdistf := flag.String("splitlist", "splitList.tsv", "a list of values for each feature that was split on")

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

	splitValueList := make(map[int][]float64, nfeatures)

	for i := 0; i < len(forest.Trees); i++ {
		splits := forest.Trees[i].GetSplits(data, caseFeatureCounts, relativeSplitCount)

		for _, split := range splits {
			featureId := data.Map[split.Feature]
			featureCounts[featureId]++ //increment the count for the total # of times the feature was a splitter

			if split.Numerical == true {
				if splitValueList[featureId] == nil {
					splitValueList[featureId] = make([]float64, 0)
				}
				splitValueList[featureId] = append(splitValueList[featureId], float64(split.Value))
			}
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

	log.Print("Outputing Split Distribution")
	splitdistfile, err := os.Create(*splitdistf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer splitdistfile.Close()
	for feature, list := range splitValueList {
		if _, err := fmt.Fprintf(splitdistfile, "%v", feature); err != nil {
			log.Fatal(err)
		}
		for _, value := range list {
			if _, err := fmt.Fprintf(splitdistfile, "\t%g", value); err != nil {
				log.Fatal(err)
			}
		}
		fmt.Fprintf(splitdistfile, "\n")

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
	relativeSplitCount.WriteTsv(rboutfile)
}
