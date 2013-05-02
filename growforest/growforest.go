package main

import (
	"encoding/json"
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
		"rface.sf", "File name to output predictor forest in sf format.")
	targetname := flag.String("target",
		"", "The row header of the target in the feature matrix.")
	imp := flag.String("importance",
		"", "File name to output importance.")
	costs := flag.String("cost",
		"", "For catagorical targets, a json string to float map of the cost of falsely identifying each catagory.")

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
	flag.BoolVar(&itter, "itterative", true, "Use an iterative search for large (n>5) catagorical fearures instead of exahustive/random.")

	var l1 bool
	flag.BoolVar(&l1, "l1", false, "Use l1 norm regression (target must be numeric).")

	flag.Parse()

	fmt.Printf("nTrees : %v\n", nTrees)

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
	fmt.Printf("nSamples : %v\n", nSamples)

	if mTry <= 0 {
		mTry = int(math.Ceil(math.Sqrt(float64(len(data.Data)))))
	}
	fmt.Printf("mTry : %v\n", mTry)

	targeti, ok := data.Map[*targetname]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	targetf := data.Data[targeti]
	if leafSize <= 0 {
		if targetf.NCats() == 0 {
			//regresion
			leafSize = 4
		} else {
			//clasification
			leafSize = 1
		}
	}
	fmt.Printf("leafSize : %v\n", leafSize)

	var target CloudForest.Target

	switch {
	case l1:
		fmt.Println("Using l1 regression.")
		target = &CloudForest.L1Target{&targetf}
	case *costs != "":
		fmt.Println("Using cost weighted classification: ", *costs)
		costmap := make(map[string]float64)
		err := json.Unmarshal([]byte(*costs), &costmap)
		if err != nil {
			log.Fatal(err)
		}

		regTarg := CloudForest.NewRegretTarget(&targetf)
		regTarg.SetCosts(costmap)
		target = regTarg

	default:
		target = &targetf
	}

	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestwriter := CloudForest.NewForestWriter(forestfile)
	forestwriter.WriteForestHeader(*targetname, nTrees)

	//fmt.Fprintf(forestfile, "FOREST=RF,TARGET=%v,NTREES=%v\n", *targetname, nTrees)

	var imppnt *[]CloudForest.RunningMean
	if *imp != "" {
		fmt.Println("Recording Importance Scores.")
		importance := make([]CloudForest.RunningMean, len(data.Data))
		imppnt = &importance
	}

	canidates := make([]int, 0, len(data.Data))
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

		tree.Grow(data, target, cases, canidates, mTry, leafSize, itter, imppnt, &l, &r)
		forestwriter.WriteTree(tree, i)
	}

	if *imp != "" {
		impfile, err := os.Create(*imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		for i, v := range *imppnt {
			fmt.Fprintf(impfile, "%v\t%v\t%v\n", data.Data[i].Name, v.Mean, v.Count)

		}
	}

}
