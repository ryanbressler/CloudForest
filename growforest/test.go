package main

import (
	"fmt"
	"log"

	"github.com/ryanbressler/CloudForest"
)

func runTest(unboostedTarget CloudForest.Feature) {

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
