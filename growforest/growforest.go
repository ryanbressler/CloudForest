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
	"regexp"
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

	rfweights := flag.String("rfweights",
		"", "For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.")

	blacklist := flag.String("blacklist",
		"", "A list of feature id's to exclude from the set of predictors.")

	var nCores int
	flag.IntVar(&nCores, "nCores", 1, "The number of cores to use.")

	var StringnSamples string
	flag.StringVar(&StringnSamples, "nSamples", "0", "The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.")

	var StringmTry string
	flag.StringVar(&StringmTry, "mTry", "0", "Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.")

	var StringleafSize string
	flag.StringVar(&StringleafSize, "leafSize", "0", "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	var shuffleRE string
	flag.StringVar(&shuffleRE, "shuffleRE", "", "A regular expression to identify features that should be shuffled.")

	var blockRE string
	flag.StringVar(&blockRE, "blockRE", "", "A regular expression to identify features that should be filtered out.")

	var includeRE string
	flag.StringVar(&includeRE, "includeRE", "", "Filter features that DON'T match this RE.")

	var nTrees int
	flag.IntVar(&nTrees, "nTrees", 100, "Number of trees to grow in the predictor.")

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

	var density bool
	flag.BoolVar(&density, "density", false, "Build density estimating trees instead of classifcation/regression trees.")

	var vet bool
	flag.BoolVar(&vet, "vet", false, "Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.")

	var evaloob bool
	flag.BoolVar(&evaloob, "evaloob", false, "Evaluate potential splitting features on OOB cases after finding split value in bag.")

	var entropy bool
	flag.BoolVar(&entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	var oob bool
	flag.BoolVar(&oob, "oob", false, "Calculate and report oob error.")

	var caseoob string
	flag.StringVar(&caseoob, "oobpreds", "", "Calculate and report oob predictions in the file specified.")

	var progress bool
	flag.BoolVar(&progress, "progress", false, "Report tree number and running oob error.")

	var adaboost bool
	flag.BoolVar(&adaboost, "adaboost", false, "Use Adaptive boosting for regression/classification.")

	var gradboost float64
	flag.Float64Var(&gradboost, "gbt", 0.0, "Use gradiant boosting with the specified learning rate.")

	var multiboost bool
	flag.BoolVar(&multiboost, "multiboost", false, "Allow multithreaded boosting which may have unexpected results. (highly experimental)")

	var nobag bool
	flag.BoolVar(&nobag, "nobag", false, "Don't bag samples for each tree.")

	var balance bool
	flag.BoolVar(&balance, "balance", false, "Balance bagging of samples by target class for unbalanced classification.")

	var balanceby string
	flag.StringVar(&balanceby, "balanceby", "", "Roughly balanced bag the target within each class of this feature.")

	var ordinal bool
	flag.BoolVar(&ordinal, "ordinal", false, "Use ordinal regression (target must be numeric).")

	var permutate bool
	flag.BoolVar(&permutate, "permute", false, "Permute the target feature (to establish random predictive power).")

	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if multiboost {
		fmt.Println("MULTIBOOST!!!!1!!!!1!!11 (things may break).")
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
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

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

	//find the target feature
	fmt.Printf("Target : %v\n", *targetname)
	targeti, ok := data.Map[*targetname]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	if blockRE != "" {
		re := regexp.MustCompile(blockRE)
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}

		}

	}

	if includeRE != "" {
		re := regexp.MustCompile(includeRE)
		for i, feature := range data.Data {
			if targeti != i && !re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}

		}
	}

	nFeatures := len(data.Data) - blacklisted - 1
	fmt.Printf("Non Target Features : %v\n", nFeatures)

	mTry := CloudForest.ParseAsIntOrFractionOfTotal(StringmTry, nFeatures)
	if mTry <= 0 {

		mTry = int(math.Ceil(math.Sqrt(float64(nFeatures))))
	}
	fmt.Printf("mTry : %v\n", mTry)

	if impute {
		fmt.Println("Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	if permutate {
		fmt.Println("Permutating target feature.")
		data.Data[targeti].Shuffle()
	}

	if shuffleRE != "" {
		re := regexp.MustCompile(shuffleRE)
		shuffled := 0
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				data.Data[i].Shuffle()
				shuffled += 1

			}

		}
		fmt.Printf("Shuffled %v features matching %v\n", shuffled, shuffleRE)
	}

	targetf := data.Data[targeti]
	unboostedTarget := targetf.Copy()

	var bSampler CloudForest.Bagger
	if balance {
		bSampler = CloudForest.NewBalancedSampler(targetf.(*CloudForest.DenseCatFeature))
	}

	if balanceby != "" {
		bSampler = CloudForest.NewSecondaryBalancedSampler(targetf.(*CloudForest.DenseCatFeature), data.Data[data.Map[balanceby]].(*CloudForest.DenseCatFeature))
		balance = true

	}

	nNonMissing := 0

	for i := 0; i < targetf.Length(); i++ {
		if !targetf.IsMissing(i) {
			nNonMissing += 1
		}

	}
	fmt.Printf("non-missing cases: %v\n", nNonMissing)

	leafSize := CloudForest.ParseAsIntOrFractionOfTotal(StringleafSize, nNonMissing)

	if leafSize <= 0 {
		if boost {
			leafSize = nNonMissing / 3
		} else if targetf.NCats() == 0 {
			//regression
			leafSize = 4
		} else {
			//classification
			leafSize = 1
		}
	}
	fmt.Printf("leafSize : %v\n", leafSize)

	//infer nSamples and mTry from data if they are 0
	nSamples := CloudForest.ParseAsIntOrFractionOfTotal(StringnSamples, nNonMissing)
	if nSamples <= 0 {
		nSamples = nNonMissing
	}
	fmt.Printf("nSamples : %v\n", nSamples)

	if progress {
		oob = true
	}
	if caseoob != "" {
		oob = true
	}
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
	var target CloudForest.Target
	if density {
		fmt.Println("Estimating Density.")
		target = &CloudForest.DensityTarget{&data.Data, nSamples}
	} else {

		switch targetf.(type) {

		case CloudForest.NumFeature:
			fmt.Println("Performing regression.")
			if l1 {
				fmt.Println("Using l1/absolute deviance error.")
				targetf = &CloudForest.L1Target{targetf.(CloudForest.NumFeature)}
			}
			if ordinal {
				fmt.Println("Using Ordinal (mode) prediction.")
				targetf = CloudForest.NewOrdinalTarget(targetf.(CloudForest.NumFeature))
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
			target = targetf

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
			case *rfweights != "":
				fmt.Println("Using rf weights: ", *rfweights)
				weightmap := make(map[string]float64)
				err := json.Unmarshal([]byte(*rfweights), &weightmap)
				if err != nil {
					log.Fatal(err)
				}

				wrfTarget := CloudForest.NewWRFTarget(targetf.(CloudForest.CatFeature), weightmap)
				targetf = wrfTarget

			case entropy:
				fmt.Println("Using entropy minimization.")
				targetf = &CloudForest.EntropyTarget{targetf.(CloudForest.CatFeature)}

			case boost:

				fmt.Println("Using Adaptive Boosting.")
				targetf = CloudForest.NewAdaBoostTarget(targetf.(CloudForest.CatFeature))

			}
			target = targetf

		}
	}

	forestfile, err := os.Create(*rf)
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestwriter := CloudForest.NewForestWriter(forestfile)

	//****************** Needed Collections and vars ******************//

	var imppnt *[]*CloudForest.RunningMean
	var mmdpnt *[]*CloudForest.RunningMean
	if *imp != "" {
		fmt.Println("Recording Importance Scores.")

		imppnt = CloudForest.NewRunningMeans(len(data.Data))
		mmdpnt = CloudForest.NewRunningMeans(len(data.Data))
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
			oobcases := make([]int, 0, nSamples)

			if nobag {
				for i := 0; i < nSamples; i++ {
					cases = append(cases, i)
				}
			}

			var depthUsed *[]int
			if mmdpnt != nil {
				du := make([]int, len(data.Data))
				depthUsed = &du
			}

			allocs := CloudForest.NewBestSplitAllocs(nSamples, targetf)
			for {
				nCases := data.Data[0].Length()
				//sample nCases case with replacement
				if !nobag {
					cases = cases[0:0]

					if balance {
						bSampler.Sample(&cases, nSamples)

					} else {
						for j := 0; len(cases) < nSamples; j++ {
							r := rand.Intn(nCases)
							if !targetf.IsMissing(r) {
								cases = append(cases, r)
							}
						}
					}

				}

				if oob || evaloob {
					ibcases := make([]bool, nCases)
					for _, v := range cases {
						ibcases[v] = true
					}
					oobcases = oobcases[0:0]
					for i, v := range ibcases {
						if !v {
							oobcases = append(oobcases, i)
						}
					}
				}

				tree.Grow(data, target, cases, canidates, oobcases, mTry, leafSize, splitmissing, vet, evaloob, imppnt, depthUsed, allocs)

				if mmdpnt != nil {
					for i, v := range *depthUsed {
						if v != 0 {
							(*mmdpnt)[i].Add(float64(v))
							(*depthUsed)[i] = 0
						}

					}
				}

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
					tree.VoteCases(data, oobVotes, oobcases)
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
		if progress {
			fmt.Printf("Model oob error after tree %v : %v\n", i, oobVotes.TallyError(unboostedTarget))
		}

	}
	if oob {
		fmt.Printf("Out of Bag Error : %v\n", oobVotes.TallyError(unboostedTarget))
	}
	if caseoob != "" {
		caseoobfile, err := os.Create(caseoob)
		if err != nil {
			log.Fatal(err)
		}
		defer caseoobfile.Close()
		for i := 0; i < unboostedTarget.Length(); i++ {
			fmt.Fprintf(caseoobfile, "%v\t%v\t%v\n", data.CaseLabels[i], oobVotes.Tally(i), unboostedTarget.GetStr(i))
		}
	}

	if *imp != "" {
		impfile, err := os.Create(*imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		for i, v := range *imppnt {
			mean, count := v.Read()
			meanMinDepth, treeCount := (*mmdpnt)[i].Read()
			fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n", data.Data[i].GetName(), mean, count, mean*float64(count)/float64(nTrees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

		}
	}

}
