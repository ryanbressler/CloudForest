package main

import (
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
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

	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	var itter bool
	flag.BoolVar(&itter, "itterative", false, "Use an iterative instead of exahustive search for splitting catagorical variables.")

	var l1 bool
	flag.BoolVar(&itter, "l1", false, "Use l1 norm regression.")

	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

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

	var target CloudForest.Target
	if l1 {
		target = &CloudForest.L1Target{&data.Data[data.Map[*targetname]]}
	} else {
		target = &data.Data[data.Map[*targetname]]
	}

	if leafSize <= 0 {
		if target.NCats() == 0 {
			//regresion
			leafSize = 4
		} else {
			//clasification
			leafSize = 1
		}
	}

	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}

	defer forestfile.Close()
	fmt.Fprintf(forestfile, "FOREST=RF,TARGET=%v,NTREES=%v\n", *targetname, nTrees)

	canidates := make([]int, 0, len(data.Data))
	targeti := data.Map[*targetname]
	for i := 0; i < len(data.Data); i++ {
		if i != targeti {
			canidates = append(canidates, i)
		}
	}
	tree := CloudForest.NewTree()
	cases := make([]int, 0, nSamples)
	l := make([]int, 0, nSamples)
	r := make([]int, 0, nSamples)
	for i := 0; i < nTrees; i++ {
		//sample nCases case with replacment
		cases = cases[0:0]
		nCases := len(data.Data[0].Missing)
		for i := 0; i < nSamples; i++ {
			cases = append(cases, rand.Intn(nCases))
		}

		tree.Grow(data, target, cases, canidates, mTry, leafSize, itter, &l, &r)
		fmt.Fprintf(forestfile, "TREE=%v\n", i)
		tree.Root.Write(forestfile, "*")
	}

}
