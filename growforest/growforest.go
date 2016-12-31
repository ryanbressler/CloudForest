package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/ryanbressler/CloudForest"
)

var (
	ok      bool
	err     error
	targeti int
	data    *CloudForest.FeatureMatrix
	trees   []*CloudForest.Tree

	nNonMissing int
	nFeatures   int
	nForest     = 1
)

func main() {
	flag.Parse()

	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if !noseed {
		rand.Seed(time.Now().UTC().UnixNano())
	}

	if testfm != "" {
		dotest = true
	}

	if multiboost {
		fmt.Println("MULTIBOOST!!!!1!!!!1!!11 (things may break).")
	}

	var boostMutex sync.Mutex
	boost := (adaboost || gradboost != 0.0)
	if boost && !multiboost {
		nCores = 1
	}

	runtime.GOMAXPROCS(nCores)

	fmt.Printf("Threads : %v\n", nCores)
	fmt.Printf("nTrees : %v\n", nTrees)

	//Parse Data
	fmt.Printf("Loading data from: %v\n", fm)
	if data, err = CloudForest.LoadAFM(fm); err != nil {
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

	blacklisted, blacklistis := generateBlacklist()

	//find the target feature
	fmt.Printf("Target : %v\n", targetname)
	if targeti, ok = data.Map[targetname]; !ok {
		log.Fatal("Target not found in data.")
	}

	regexBlacklist(blacklistis, blacklisted)
	regexWhitelist(blacklistis, blacklisted)

	nFeatures = len(data.Data) - blacklisted - 1
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
		fmt.Println("Permuting target feature.")
		data.Data[targeti].Shuffle()
	}

	regexShuffle()

	targetf := data.Data[targeti]
	unboostedTarget := targetf.Copy()

	//TODO:: if neither of these are set, we  have a nil bSampler?
	var bSampler CloudForest.Bagger
	if balance {
		bSampler = CloudForest.NewBalancedSampler(targetf.(*CloudForest.DenseCatFeature))
	}

	if balanceby != "" {
		bSampler = CloudForest.NewSecondaryBalancedSampler(
			targetf.(*CloudForest.DenseCatFeature),
			data.Data[data.Map[balanceby]].(*CloudForest.DenseCatFeature))
		balance = true
	}

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

	if progress || caseoob != "" {
		oob = true
	}

	oobVoteTallier(targetf)

	target := getTarget(targetf)

	var forestwriter *CloudForest.ForestWriter
	if rf != "" {
		forestfile, err := os.Create(rf)
		if err != nil {
			log.Fatal(err)
		}
		defer forestfile.Close()
		forestwriter = CloudForest.NewForestWriter(forestfile)

		switch target.(type) {
		case CloudForest.TargetWithIntercept:
			forestwriter.WriteForestHeader(0, targetname,
				target.(CloudForest.TargetWithIntercept).Intercept())
		}
	}

	runAce(blacklistis)

	trees = make([]*CloudForest.Tree, 0, nTrees)

	recordScores()

	//****************** Good Stuff Stars Here ;) ******************//
	trainingStart := time.Now()

	for foresti := 0; foresti < nForest; foresti++ {
		var treesStarted, treesFinished int
		treesStarted = nCores

		var recordingTreeMutex sync.Mutex
		var waitGroup sync.WaitGroup

		for core := 0; core < nCores; core++ {

			waitGroup.Add(1)

			go func(foresti int) {
				defer waitGroup.Done()

				weight := -1.0
				canidates := make([]int, 0, len(data.Data))
				for i := 0; i < len(data.Data); i++ {
					if i != targeti && !blacklistis[i] {
						canidates = append(canidates, i)
					}
				}

				tree := CloudForest.NewTree()
				tree.Target = targetname

				cases := make([]int, 0, nNonMissing)
				oobcases := make([]int, 0, nNonMissing)

				if nobag {
					for i := 0; i < nNonMissing; i++ {
						if !targetf.IsMissing(i) {
							cases = append(cases, i)
						}
					}
				}

				depthUsed := []int{}
				if mmdpnt != nil {
					depthUsed = make([]int, len(data.Data))
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

					if nobag && nSamples != nCases {
						cases = cases[0:0]
						for i := 0; i < nCases; i++ {
							if !targetf.IsMissing(i) {
								cases = append(cases, i)
							}
						}
						CloudForest.SampleFirstN(&cases, &cases, nSamples, 0)
					}

					addOOB(cases, oobcases)

					if jungle {
						tree.GrowJungle(data, target, cases, canidates,
							oobcases, mTry, leafSize, maxDepth, splitmissing,
							force, vet, evaloob, extra, imppnt, &depthUsed, allocs)

					} else {
						tree.Grow(data, target, cases, canidates,
							oobcases, mTry, leafSize, maxDepth, splitmissing,
							force, vet, evaloob, extra, imppnt, &depthUsed, allocs)
					}

					if mmdpnt != nil {
						for i, v := range depthUsed {
							if v != 0 {
								(*mmdpnt)[i].Add(float64(v))
								depthUsed[i] = 0
							}

						}
					}

					if boost {
						boostMutex.Lock()
						ls, ps := tree.Partition(data)
						weight = targetf.(CloudForest.BoostingTarget).Boost(ls, ps)
						boostMutex.Unlock()
						if math.IsInf(weight, 1) {
							fmt.Printf("Boosting Reached Weight of %v\n", weight)
							break
						}
						tree.Weight = weight
					}

					if oob && foresti == nForest-1 {
						tree.VoteCases(data, oobVotes, oobcases)
					}

					////////////// Lock mutext to ouput tree ////////
					if nCores > 1 {
						recordingTreeMutex.Lock()
					}

					if forestwriter != nil && foresti == nForest-1 {
						forestwriter.WriteTree(tree, treesFinished)
					}

					addScikit(tree)

					if dotest && foresti == nForest-1 {
						trees = append(trees, tree)

						if treesStarted < nTrees-1 {
							tree = CloudForest.NewTree()
							tree.Target = targetname
						}
					}

					if progress {
						treesFinished++
						fmt.Printf("Model oob error after tree %v : %v\n", treesFinished,
							oobVotes.TallyError(unboostedTarget))
					}

					if treesStarted < nTrees {
						treesStarted++
					} else {
						if nCores > 1 {
							recordingTreeMutex.Unlock()
						}
						break
					}

					if nCores > 1 {
						recordingTreeMutex.Unlock()
					}
				}
			}(foresti)

		}

		waitGroup.Wait()

		//TODO:: Come up with a better name
		acer(mTry, blacklistis, foresti)
	}

	fmt.Printf("Total training time (seconds): %v\n", time.Now().Sub(trainingStart).Seconds())

	writeScikit()

	writeOOB(unboostedTarget)

	writeImportance()

	runTest(unboostedTarget)
}
