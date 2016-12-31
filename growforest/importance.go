package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/ryanbressler/CloudForest"
	"github.com/ryanbressler/CloudForest/stats"
)

//****************** Setup For ACE ********************************//
var aceImps [][]float64
var firstace int

var imppnt *[]*CloudForest.RunningMean
var mmdpnt *[]*CloudForest.RunningMean

func runAce(blacklistis []bool) {
	firstace = len(data.Data)
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
}

func recordScores() {
	if imp != "" {
		fmt.Println("Recording Importance Scores.")
		imppnt = CloudForest.NewRunningMeans(len(data.Data))
		mmdpnt = CloudForest.NewRunningMeans(len(data.Data))
	} else if ace > 0 {
		imppnt = CloudForest.NewRunningMeans(len(data.Data))
	}
}

func acer(mTry int, blacklistis []bool, foresti int) {
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

func writeImportance() {
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
			return
		}

		//Write standard importance file
		for i, v := range *imppnt {
			mean, count := v.Read()
			meanMinDepth, treeCount := (*mmdpnt)[i].Read()
			fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n",
				data.Data[i].GetName(), mean, count,
				mean*float64(count)/float64(nTrees),
				mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

		}
	}

}
