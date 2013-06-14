package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
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
		"", "For categorical targets, a json string to float map of the cost of falsely identifying each category.")

	blacklist := flag.String("blacklist",
		"", "A list of feature id's to exclude from the set of predictors.")

	var nCores int
	flag.IntVar(&nCores, "nCores", 1, "The number of cores to use.")

	var nSamples int
	flag.IntVar(&nSamples, "nSamples", 0, "The number of cases to sample (with replacement) for each tree grow. If <=0 set to total number of cases")

	var leafSize int
	flag.IntVar(&leafSize, "leafSize", 0, "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	var nTrees int
	flag.IntVar(&nTrees, "nTrees", 100, "Number of trees to grow in the predictor.")

	var mTry int
	flag.IntVar(&mTry, "mTry", 0, "Number of candidate features for each split. Inferred to ceil(sqrt(nFeatures)) if <=0.")

	var nContrasts int
	flag.IntVar(&nContrasts, "nContrasts", 0, "The number of randomized artificial contrast features to include in the feature matrix.")

	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	var contrastAll bool
	flag.BoolVar(&contrastAll, "contrastall", false, "Include a shuffled artificial contrast copy of every feature.")

	var impute bool
	flag.BoolVar(&impute, "impute", false, "Impute missing values to feature mean/mode before growth.")

	var splitmissing bool
	flag.BoolVar(&splitmissing, "splitmissing", false, "Split missing values onto a third branch at each node (experimental).")

	var l1 bool
	flag.BoolVar(&l1, "l1", false, "Use l1 norm regression (target must be numeric).")

	var entropy bool
	flag.BoolVar(&entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	var oob bool
	flag.BoolVar(&oob, "oob", false, "Calculte and report oob error.")

	var adaboost bool
	flag.BoolVar(&adaboost, "adaboost", false, "Use Adaptive boosting for regresion/classification.")

	var gradboost float64
	flag.Float64Var(&gradboost, "gradboost", 0.0, "Use gradiant boosting with the specified learning rate.")

	var multiboost bool
	flag.BoolVar(&multiboost, "multiboost", false, "Allow multithreaded boosting which msy have unexpected results. (highly experimental)")

	var nobag bool
	flag.BoolVar(&nobag, "nobag", false, "Don't bag samples for each tree.")

	var ordinal bool
	flag.BoolVar(&ordinal, "ordinal", false, "Use ordinal regression (target must be numeric).")

	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	var boostMutex sync.Mutex
	boost := (adaboost || gradboost != 0.0)
	if boost && !multiboost {
		nCores = 1
	}

	if nCores > 1 {

		runtime.GOMAXPROCS(nCores)
	}
	fmt.Printf("Threads : %v\n", nCores)
	fmt.Printf("nTrees : %v\n", nTrees)
	//Parse Data
	fmt.Printf("Loading data from: %v\n", *fm)
	datafile, err := os.Open(*fm)
	if err != nil {
		log.Fatal(err)
	}

	data := CloudForest.ParseAFM(datafile)
	datafile.Close()

	//infer nSamples and mTry from data if they are 0
	if nSamples <= 0 {
		nSamples = data.Data[0].Length()
	}
	fmt.Printf("nSamples : %v\n", nSamples)

	if nContrasts > 0 {
		fmt.Printf("Adding %v Random Contrasts\n", nContrasts)
		data.AddContrasts(nContrasts)
	}
	if contrastAll {
		fmt.Printf("Adding Random Contrasts for All Features.\n")
		data.ContrastAll()
	}

	blacklisted := 0
	blacklistis := make([]bool, len(data.Data))
	if *blacklist != "" {
		fmt.Printf("Loading blacklist from: %v\n", *blacklist)
		blackfile, err := os.Open(*blacklist)
		if err != nil {
			log.Fatal(err)
		}
		tsv := csv.NewReader(blackfile)
		tsv.Comma = '\t'
		for {
			id, err := tsv.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal(err)
			}
			i, ok := data.Map[id[0]]
			if !ok {
				fmt.Printf("Ignoring blacklist feature not found in data: %v\n", id[0])
				continue
			}
			if !blacklistis[i] {
				blacklisted += 1
				blacklistis[i] = true
			}

		}
		blackfile.Close()

	}

	nFeatures := len(data.Data) - blacklisted
	fmt.Printf("nFeatures : %v\n", nFeatures)
	if mTry <= 0 {

		mTry = int(math.Ceil(math.Sqrt(float64(nFeatures))))
	}
	fmt.Printf("mTry : %v\n", mTry)

	if impute {
		fmt.Println("Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	//find the target feature
	targeti, ok := data.Map[*targetname]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	targetf := data.Data[targeti]
	if leafSize <= 0 {
		if boost {
			leafSize = targetf.Length() / 3
		} else if targetf.NCats() == 0 {
			//regression
			leafSize = 4
		} else {
			//classification
			leafSize = 1
		}
	}
	fmt.Printf("leafSize : %v\n", leafSize)

	var oobVotes CloudForest.VoteTallyer
	if oob {
		fmt.Println("Recording oob error.")
		if targetf.NCats() == 0 {
			//regression
			oobVotes = CloudForest.NewNumBallotBox(data.Data[0].Length())
		} else {
			//classification
			oobVotes = CloudForest.NewCatBallotBox(data.Data[0].Length())
		}
	}

	//****** Set up Target for Alternative Impurity  if needed *******//

	switch targetf.(type) {
	case CloudForest.NumFeature:
		//BUG(ryan): test if these targets actually stack.
		if l1 {
			fmt.Println("Using l1/absolute deviance error.")
			targetf = &CloudForest.L1Target{targetf.(CloudForest.NumFeature)}
		}
		if ordinal {
			fmt.Println("Using Ordinal (mode) prediction.")
			targetf = &CloudForest.OrdinalTarget{targetf.(CloudForest.NumFeature)}
		}
		switch {
		case gradboost != 0.0:
			fmt.Println("Using Gradiant Boosting.")
			targetf = &CloudForest.GradBoostTarget{targetf.(CloudForest.NumFeature), gradboost}

		case adaboost:
			fmt.Println("Using Numeric Adaptive Boosting.")
			//BUG(ryan): gradiant boostign should expose learning rate.
			targetf = CloudForest.NewNumAdaBoostTarget(targetf.(CloudForest.NumFeature))
		}

	case CloudForest.CatFeature:
		fmt.Println("Performing classification.")
		switch {
		case *costs != "":
			fmt.Println("Using missclasification costs: ", *costs)
			costmap := make(map[string]float64)
			err := json.Unmarshal([]byte(*costs), &costmap)
			if err != nil {
				log.Fatal(err)
			}

			regTarg := CloudForest.NewRegretTarget(targetf.(CloudForest.CatFeature))
			regTarg.SetCosts(costmap)
			targetf = regTarg

		case entropy:
			fmt.Println("Using entropy minimization.")
			targetf = &CloudForest.EntropyTarget{targetf.(CloudForest.CatFeature)}

		case boost:

			fmt.Println("Using Adaptive Boosting.")
			targetf = CloudForest.NewAdaBoostTarget(targetf.(CloudForest.CatFeature))

		}

	}

	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestwriter := CloudForest.NewForestWriter(forestfile)
	//forestwriter.WriteForestHeader(*targetname, nTrees)

	//****************** Needed Collections and vars ******************//

	var imppnt *[]*CloudForest.RunningMean
	if *imp != "" {
		fmt.Println("Recording Importance Scores.")

		imppnt = CloudForest.NewRunningMeans(len(data.Data))
	}

	treechan := make(chan *CloudForest.Tree, 0)

	//****************** Good Stuff Stars Here ;) ******************//
	for core := 0; core < nCores; core++ {
		go func() {
			weight := -1.0
			canidates := make([]int, 0, len(data.Data))
			for i := 0; i < len(data.Data); i++ {
				if i != targeti && !blacklistis[i] {
					canidates = append(canidates, i)
				}
			}
			tree := CloudForest.NewTree()
			tree.Target = *targetname
			cases := make([]int, 0, nSamples)
			if nobag {
				for i := 0; i < nSamples; i++ {
					cases = append(cases, i)
				}
			}

			allocs := CloudForest.NewBestSplitAllocs(nSamples, targetf)
			for {
				nCases := data.Data[0].Length()
				//sample nCases case with replacement
				if !nobag {
					cases = cases[0:0]

					for j := 0; j < nSamples; j++ {
						cases = append(cases, rand.Intn(nCases))
					}
				}

				tree.Grow(data, targetf, cases, canidates, mTry, leafSize, splitmissing, imppnt, allocs)

				if boost {
					boostMutex.Lock()
					weight = targetf.(CloudForest.BoostingTarget).Boost(tree.Partition(data))
					boostMutex.Unlock()
					if weight == math.Inf(1) {
						fmt.Printf("Boosting Reached Weight of %v\n", weight)
						close(treechan)
						break
					}

					tree.Weight = weight
				}

				if oob {
					ibcases := make([]bool, nCases)
					for _, v := range cases {
						ibcases[v] = true
					}
					cases = cases[0:0]
					for i, v := range ibcases {
						if !v {
							cases = append(cases, i)
						}
					}

					tree.VoteCases(data, oobVotes, cases)
				}

				treechan <- tree
				tree = <-treechan
			}
		}()

	}

	for i := 0; i < nTrees; i++ {
		tree := <-treechan
		if tree == nil {
			break
		}

		forestwriter.WriteTree(tree, i)
		if i < nTrees-1 {
			treechan <- tree
		}

	}
	if oob {
		fmt.Printf("Out of Bag Error : %v\n", oobVotes.TallyError(targetf))
	}

	if *imp != "" {
		impfile, err := os.Create(*imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		for i, v := range *imppnt {
			mean, count := v.Read()
			fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\n", data.Data[i].GetName(), mean, count, mean*float64(count)/float64(nTrees))

		}
	}

}
