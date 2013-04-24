package main

import (
	"flag"
	"github.com/ryanbressler/CloudForest"
	"log"
	"math"
	"os"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing test data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest as outputed by rf-ace")
	targetname := flag.String("target",
		"", "The row header of the target in the feature matrix.")
	flag.Parse()

	datafile, err := os.Open(*fm)
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := CloudForest.ParseAFM(datafile)

	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}

	defer forestfile.Close()
	nSamples := len(data.Data[0].Missing)
	mTry := int(math.Ceil(math.Sqrt(float64(len(data.Data)))))
	leafSize := 1
	nTrees := 4
	target := &data.Data[data.Map[*targetname]]
	forrest := CloudForest.GrowRandomForest(data, target, nSamples, mTry, nTrees, leafSize)
	forrest.SavePredictor(forestfile)
}
