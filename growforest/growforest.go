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
	flag.IntVar(&mTry, "mTry", 0, "Number of candidate features for each split. Inferred to ceil(swrt(nFeatures)) if <=0.")

	var nContrasts int
	flag.IntVar(&nContrasts, "nContrasts", 0, "The number of randomized artificial contrast features to include in the feature matrix.")

	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	var contrastAll bool
	flag.BoolVar(&contrastAll, "contrastall", false, "Include a shuffled artificial contrast copy of every feature.")

	var impute bool
	flag.BoolVar(&impute, "impute", false, "Impute missing values to feature mean/mode instead of filtering them out when splitting.")

	var splitmissing bool
	flag.BoolVar(&splitmissing, "splitmissing", false, "Split missing values onto a third branch at each node (experimental).")

	var l1 bool
	flag.BoolVar(&l1, "l1", false, "Use l1 norm regression (target must be numeric).")

	var entropy bool
	flag.BoolVar(&entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	var oob bool
	flag.BoolVar(&oob, "oob", false, "Calculte and report oob error.")

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

	if nCores > 1 {
		runtime.GOMAXPROCS(nCores)
	}
	//Parse Data
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

	if mTry <= 0 {
		mTry = int(math.Ceil(math.Sqrt(float64(len(data.Data) - blacklisted))))
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
		if targetf.NCats() == 0 {
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
			oobVotes = CloudForest.NewNumBallotBox(len(data.Data[0].Missing))
		} else {
			//classification
			oobVotes = CloudForest.NewCatBallotBox(len(data.Data[0].Missing))
		}
	}

	//****** Set up Target for Alternative Impurity  if needed *******//
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

	case entropy:
		fmt.Println("Using entropy minimizing classification.")
		target = &CloudForest.EntropyTarget{&targetf}

	default:
		target = &targetf
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
			canidates := make([]int, 0, len(data.Data))
			for i := 0; i < len(data.Data); i++ {
				if i != targeti && !blacklistis[i] {
					canidates = append(canidates, i)
				}
			}
			tree := CloudForest.NewTree()
			tree.Target = targetf.Name
			cases := make([]int, 0, nSamples)
			allocs := CloudForest.NewBestSplitAllocs(nSamples, target)
			for i := 0; i < nTrees; i++ {
				//sample nCases case with replacement
				cases = cases[0:0]
				nCases := len(data.Data[0].Missing)
				for j := 0; j < nSamples; j++ {
					cases = append(cases, rand.Intn(nCases))
				}

				tree.Grow(data, target, cases, canidates, mTry, leafSize, splitmissing, imppnt, allocs)
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
		forestwriter.WriteTree(tree, i)
		if i < nTrees-1 {
			treechan <- tree
		}

	}
	if oob {
		fmt.Printf("Oib Error : %v\n", oobVotes.TallyError(&targetf))
	}

	if *imp != "" {
		impfile, err := os.Create(*imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		for i, v := range *imppnt {
			mean, count := v.Read()
			fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\n", data.Data[i].Name, mean, count, mean*float64(count))

		}
	}

}
