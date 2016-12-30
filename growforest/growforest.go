package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/ryanbressler/CloudForest"
	"github.com/ryanbressler/CloudForest/stats"
)

func main() {
	flag.Parse()

	nForest := 1

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

	if nCores > 1 {

		runtime.GOMAXPROCS(nCores)
	}
	fmt.Printf("Threads : %v\n", nCores)
	fmt.Printf("nTrees : %v\n", nTrees)
	//Parse Data
	fmt.Printf("Loading data from: %v\n", fm)
	data, err := CloudForest.LoadAFM(fm)
	if err != nil {
		log.Fatal(err)
	}

	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
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
	if blacklist != "" {
		fmt.Printf("Loading blacklist from: %v\n", blacklist)
		blackfile, err := os.Open(blacklist)
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
	fmt.Printf("Target : %v\n", targetname)
	targeti, ok := data.Map[targetname]
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
		fmt.Println("Permuting target feature.")
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
		target = &CloudForest.DensityTarget{&data.Data, nNonMissing}
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
				fmt.Println("Using Gradient Boosting.")
				targetf = CloudForest.NewGradBoostTarget(targetf.(CloudForest.NumFeature), gradboost)

			case adaboost:
				fmt.Println("Using Numeric Adaptive Boosting.")
				targetf = CloudForest.NewNumAdaBoostTarget(targetf.(CloudForest.NumFeature))
			}
			target = targetf

		case CloudForest.CatFeature:
			fmt.Printf("Performing classification with %v categories.\n", targetf.NCats())
			switch {
			case NP:
				fmt.Printf("Performing Approximate Neyman-Pearson Classification with constrained false \"%v\".\n", NP_pos)
				fmt.Printf("False %v constraint: %v, constraint weight: %v.\n", NP_pos, NP_a, NP_k)
				targetf = CloudForest.NewNPTarget(targetf.(CloudForest.CatFeature), NP_pos, NP_a, NP_k)
			case costs != "":
				fmt.Println("Using misclassification costs: ", costs)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(costs), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				regTarg := CloudForest.NewRegretTarget(targetf.(CloudForest.CatFeature))
				regTarg.SetCosts(costmap)
				targetf = regTarg
			case dentropy != "":
				fmt.Println("Using entropy with disutilities: ", dentropy)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(dentropy), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				deTarg := CloudForest.NewDEntropyTarget(targetf.(CloudForest.CatFeature))
				deTarg.SetCosts(costmap)
				targetf = deTarg
			case adacosts != "":
				fmt.Println("Using cost sensative AdaBoost costs: ", adacosts)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(adacosts), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				actarget := CloudForest.NewAdaCostTarget(targetf.(CloudForest.CatFeature))
				actarget.SetCosts(costmap)
				targetf = actarget

			case rfweights != "":
				fmt.Println("Using rf weights: ", rfweights)
				weightmap := make(map[string]float64)
				err := json.Unmarshal([]byte(rfweights), &weightmap)
				if err != nil {
					log.Fatal(err)
				}

				wrfTarget := CloudForest.NewWRFTarget(targetf.(CloudForest.CatFeature), weightmap)
				targetf = wrfTarget

			case entropy:
				fmt.Println("Using entropy minimization.")
				targetf = &CloudForest.EntropyTarget{targetf.(CloudForest.CatFeature)}

			case adaboost:

				fmt.Println("Using Adaptive Boosting.")
				targetf = CloudForest.NewAdaBoostTarget(targetf.(CloudForest.CatFeature))

			case hellinger:
				fmt.Println("Using Hellinger Distance with postive class:", positive)
				targetf = CloudForest.NewHDistanceTarget(targetf.(CloudForest.CatFeature), positive)

			case gradboost != 0.0:
				fmt.Println("Using Gradient Boosting Classification with postive class:", positive)
				targetf = CloudForest.NewGradBoostClassTarget(targetf.(CloudForest.CatFeature), gradboost, positive)

			}

			if unlabeled != "" {
				fmt.Println("Using traduction forests with unlabeled class: ", unlabeled)
				targetf = CloudForest.NewTransTarget(targetf.(CloudForest.CatFeature), &data.Data, unlabeled, trans_alpha, trans_beta, nNonMissing)

			}
			target = targetf

		}
	}

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
			forestwriter.WriteForestHeader(0, targetname, target.(CloudForest.TargetWithIntercept).Intercept())
		}
	}
	//****************** Setup For ACE ********************************//
	var aceImps [][]float64
	firstace := len(data.Data)

	if ace > 0 {

		fmt.Printf("Performing ACE analysis with %v forests/permutations.\n", ace)

		data.ContrastAll()

		for i := 0; i < firstace; i++ {
			blacklistis = append(blacklistis, blacklistis[i])
		}
		blacklistis[targeti+firstace] = true

		aceImps = make([][]float64, len(data.Data))
		for i := 0; i < len(data.Data); i++ {
			aceImps[i] = make([]float64, ace)
		}
		nForest = ace
		if cutoff > 0 {
			nForest++
		}
	}

	//****************** Needed Collections and vars ******************//
	var trees []*CloudForest.Tree
	trees = make([]*CloudForest.Tree, 0, nTrees)

	var imppnt *[]*CloudForest.RunningMean
	var mmdpnt *[]*CloudForest.RunningMean
	if imp != "" {
		fmt.Println("Recording Importance Scores.")

		imppnt = CloudForest.NewRunningMeans(len(data.Data))
		mmdpnt = CloudForest.NewRunningMeans(len(data.Data))
	} else if ace > 0 {
		imppnt = CloudForest.NewRunningMeans(len(data.Data))
	}

	var scikikittrees []CloudForest.ScikitTree

	if scikitforest != "" {
		scikikittrees = make([]CloudForest.ScikitTree, 0, nTrees)
	}

	//****************** Good Stuff Stars Here ;) ******************//

	trainingStart := time.Now()

	for foresti := 0; foresti < nForest; foresti++ {
		var treesStarted, treesFinished int
		treesStarted = nCores
		var recordingTree sync.Mutex
		var waitGroup sync.WaitGroup

		waitGroup.Add(nCores)
		treechan := make(chan *CloudForest.Tree, 0)
		//fmt.Println("forest ", foresti)
		//Grow a single forest on nCores
		for core := 0; core < nCores; core++ {

			grow := func() {
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

					if nobag && nSamples != nCases {
						cases = cases[0:0]
						for i := 0; i < nCases; i++ {
							if !targetf.IsMissing(i) {
								cases = append(cases, i)
							}
						}
						CloudForest.SampleFirstN(&cases, &cases, nSamples, 0)

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

					if jungle {
						tree.GrowJungle(data, target, cases, canidates, oobcases, mTry, leafSize, maxDepth, splitmissing, force, vet, evaloob, extra, imppnt, depthUsed, allocs)

					} else {
						tree.Grow(data, target, cases, canidates, oobcases, mTry, leafSize, maxDepth, splitmissing, force, vet, evaloob, extra, imppnt, depthUsed, allocs)
					}
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
						ls, ps := tree.Partition(data)
						weight = targetf.(CloudForest.BoostingTarget).Boost(ls, ps)
						boostMutex.Unlock()
						if weight == math.Inf(1) {
							fmt.Printf("Boosting Reached Weight of %v\n", weight)
							close(treechan)
							break
						}

						tree.Weight = weight
					}

					if oob && foresti == nForest-1 {
						tree.VoteCases(data, oobVotes, oobcases)
					}

					////////////// Lock mutext to ouput tree ////////
					if nCores > 1 {
						recordingTree.Lock()
					}

					if forestwriter != nil && foresti == nForest-1 {
						forestwriter.WriteTree(tree, treesFinished)
					}

					if scikitforest != "" {
						skt := CloudForest.NewScikitTree(nFeatures)
						CloudForest.BuildScikitTree(0, tree.Root, skt)
						scikikittrees = append(scikikittrees, *skt)
					}

					if dotest && foresti == nForest-1 {
						trees = append(trees, tree)

						if treesStarted < nTrees-1 {
							//newtree := new(CloudForest.Tree)
							tree = CloudForest.NewTree()
							tree.Target = targetname
						}
					}
					if progress {
						treesFinished++
						fmt.Printf("Model oob error after tree %v : %v\n", treesFinished, oobVotes.TallyError(unboostedTarget))
					}
					if treesStarted < nTrees {
						treesStarted++
					} else {
						if nCores > 1 {
							recordingTree.Unlock()
							waitGroup.Done()
						}
						break

					}
					if nCores > 1 {
						recordingTree.Unlock()
					}
					//////// Unlock //////////////////////////
					// treechan <- tree
					// tree = <-treechan
				}
			}

			if nCores > 1 {
				go grow()
			} else {
				grow()
			}

		}
		if nCores > 1 {
			waitGroup.Wait()
		}
		// for i := 0; i < nTrees; i++ {
		// 	tree := <-treechan
		// 	if tree == nil {
		// 		break
		// 	}
		// 	if forestwriter != nil && foresti == nForest-1 {
		// 		forestwriter.WriteTree(tree, i)
		// 	}

		// 	if dotest && foresti == nForest-1 {
		// 		trees = append(trees, tree)

		// 		if i < nTrees-1 {
		// 			//newtree := new(CloudForest.Tree)
		// 			treechan <- CloudForest.NewTree()
		// 		}
		// 	} else {
		// 		if i < nTrees-1 {
		// 			treechan <- tree
		// 		}
		// 	}
		// 	if progress {
		// 		fmt.Printf("Model oob error after tree %v : %v\n", i, oobVotes.TallyError(unboostedTarget))
		// 	}

		// }
		//Single forest growth is over.

		//Record importance scores from this forest for ace
		if ace > 0 && (cutoff == 0.0 || foresti < nForest-1) {
			if foresti < nForest-1 {
				fmt.Printf("Finished ACE forest %v.\n", foresti)
			}
			//Record Importance scores
			for i := 0; i < len(data.Data); i++ {
				mean, count := (*imppnt)[i].Read()
				aceImps[i][foresti] = mean * float64(count) / float64(nTrees)
			}

			//Reset importance scores
			imppnt = CloudForest.NewRunningMeans(len(data.Data))

			//Reshuffle contrast features
			for i := firstace; i < len(data.Data); i++ {
				if !blacklistis[i] {
					data.Data[i].Shuffle()
				}
			}

			if cutoff > 0 && foresti == nForest-2 {
				sigcount := 0
				for i := range blacklistis {

					if i < firstace && !blacklistis[i] {
						p, _, _, m := stats.Ttest(&aceImps[i], &aceImps[i+firstace])
						if p < cutoff && m > 0.0 && i != targeti {
							blacklistis[i] = false
							sigcount++
						} else {
							blacklistis[i] = true
						}
					}
					if i >= firstace {
						blacklistis[i] = true
					}

				}
				mTry = CloudForest.ParseAsIntOrFractionOfTotal(StringmTry, sigcount)
				if mTry <= 0 {

					mTry = int(math.Ceil(math.Sqrt(float64(sigcount))))
				}
				fmt.Printf("Growing non-ACE forest with %v features with p-value < %v.\nmTry: %v\n", sigcount, cutoff, mTry)
			}
		}
	}

	trainingEnd := time.Now()
	fmt.Printf("Total training time (seconds): %v\n", trainingEnd.Sub(trainingStart).Seconds())

	if scikitforest != "" {
		skfile, err := os.Create(scikitforest)
		if err != nil {
			log.Fatal(err)
		}
		defer skfile.Close()
		skencoder := json.NewEncoder(skfile)
		err = skencoder.Encode(scikikittrees)
		if err != nil {
			log.Fatal(err)
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

	if imp != "" {

		impfile, err := os.Create(imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		if ace > 0 {

			for i := 0; i < firstace; i++ {

				p, _, _, m := stats.Ttest(&aceImps[i], &aceImps[i+firstace])

				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\n", targetname, data.Data[i].GetName(), p, m)

			}
		} else {
			//Write standard importance file
			for i, v := range *imppnt {
				mean, count := v.Read()
				meanMinDepth, treeCount := (*mmdpnt)[i].Read()
				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n", data.Data[i].GetName(), mean, count, mean*float64(count)/float64(nTrees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

			}
		}
	}

	if dotest {
		var bb CloudForest.VoteTallyer

		testdata := data
		testtarget := unboostedTarget
		if testfm != "" {
			var err error
			testdata, err = CloudForest.LoadAFM(testfm)
			if err != nil {
				log.Fatal(err)
			}
			targeti, ok = testdata.Map[targetname]
			if !ok {
				log.Fatal("Target not found in test data.")
			}
			testtarget = testdata.Data[targeti]

			for _, tree := range trees {

				tree.StripCodes()

			}
		}

		if unboostedTarget.NCats() == 0 {
			//regression
			bb = CloudForest.NewNumBallotBox(testdata.Data[0].Length())
		} else {
			//classification
			bb = CloudForest.NewCatBallotBox(testdata.Data[0].Length())
		}

		for _, tree := range trees {
			tree.Vote(testdata, bb)
		}

		fmt.Printf("Error: %v\n", bb.TallyError(testtarget))

		if testtarget.NCats() != 0 {
			falsesbypred := make([]int, testtarget.NCats())
			predtotals := make([]int, testtarget.NCats())

			truebytrue := make([]int, testtarget.NCats())
			truetotals := make([]int, testtarget.NCats())

			correct := 0
			nas := 0
			length := testtarget.Length()
			for i := 0; i < length; i++ {
				truei := testtarget.(*CloudForest.DenseCatFeature).Geti(i)
				truetotals[truei]++
				pred := bb.Tally(i)
				if pred == "NA" {
					nas++
				} else {
					predi := testtarget.(*CloudForest.DenseCatFeature).CatToNum(pred)
					predtotals[predi]++
					if pred == testtarget.GetStr(i) {
						correct++
						truebytrue[truei]++
					} else {

						falsesbypred[predi]++
					}
				}

			}
			fmt.Printf("Classified: %v / %v = %v\n", correct, length, float64(correct)/float64(length))
			for i, v := range testtarget.(*CloudForest.DenseCatFeature).Back {
				fmt.Printf("Label %v Percision (Actuall/Predicted): %v / %v = %v\n", v, falsesbypred[i], predtotals[i], float64(falsesbypred[i])/float64(predtotals[i]))
				falses := truetotals[i] - truebytrue[i]
				fmt.Printf("Label %v Missed/Actuall Rate: %v / %v = %v\n", v, falses, truetotals[i], float64(falses)/float64(truetotals[i]))

			}
			if nas != 0 {
				fmt.Printf("Couldn't predict %v cases due to missing values.\n", nas)
			}
		}

	}

}
