package main

import (
	"flag"
	"github.com/ryanbressler/CloudForest"
	"log"
	"math"
	"os"
)

func main() {
	fm := flag.String("train",
		"featurematrix.afm", "AFM formated feature matrix containing training data.")
	rf := flag.String("rfpred",
		"rface.sf", "File name to output predictor in rf-aces sf format.")
	targetname := flag.String("target",
		"", "The row header of the target in the feature matrix.")

	var nSamples int
	flag.IntVar(&nSamples, "nSamples", 0, "The number of cases to sample (with replacment) for each tree grow. If <=0 set to total number of cases")

	var leafSize int
	flag.IntVar(&leafSize, "leafSize", 0, "The minimum number of cases on a leaf node. If <=0 will be infered to 1 for clasification 4 for regression.")

	var nTrees int
	flag.IntVar(&nTrees, "nTrees", 100, "Number of trees to grow in the predictor.")

	var mTry int
	flag.IntVar(&mTry, "mTry", 0, "Number of canidate features for each split. Infered to ceil(swrt(nFeatures)) if <=0.")
	flag.Parse()

	datafile, err := os.Open(*fm)
	if err != nil {
		log.Fatal(err)
	}

	data := CloudForest.ParseAFM(datafile)
	datafile.Close()

	//infer nSamples and mTry from data if they are 0
	if nSamples <= 0 {
		nSamples = len(data.Data[0].Missing)
	}
	if mTry <= 0 {
		mTry = int(math.Ceil(math.Sqrt(float64(len(data.Data)))))
	}
	target := &data.Data[data.Map[*targetname]]

	if leafSize <= 0 {
		switch target.Numerical {
		case true:
			leafSize = 4
		case false:
			leafSize = 1
		}
	}
	//create output file now to make sure it is writeable before doing long computations. 
	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}

	defer forestfile.Close()

	forrest := CloudForest.GrowRandomForest(data, target, nSamples, mTry, nTrees, leafSize)
	forrest.SavePredictor(forestfile)
}
