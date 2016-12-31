package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ryanbressler/CloudForest"
)

var oobVotes CloudForest.VoteTallyer

func oobVoteTallier(targetf CloudForest.Target) {
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
}

func addOOB(cases []int, oobcases []int) {
	if oob || evaloob {
		ibcases := make([]bool, data.Data[0].Length())
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
}

func writeOOB(unboostedTarget CloudForest.Feature) {
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
}
