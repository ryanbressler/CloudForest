package main

import (
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest.")
	predfn := flag.String("preds",
		"", "The name of a file to write the predictions into.")
	votefn := flag.String("votes",
		"", "The name of a file to write categorical vote totals to.")
	var num bool
	flag.BoolVar(&num, "mean", false, "Force numeric (mean) voting.")
	var sum bool
	flag.BoolVar(&sum, "sum", false, "Force numeric sum voting (for gradient boosting etc).")
	var expit bool
	flag.BoolVar(&expit, "expit", false, "Expit (inverst logit) transform data (for gradient boosting classification).")
	var cat bool
	flag.BoolVar(&cat, "mode", false, "Force categorical (mode) voting.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestreader := CloudForest.NewForestReader(forestfile)
	forest, err := forestreader.ReadForest()
	if err != nil {
		log.Fatal(err)
	}

	var predfile *os.File
	if *predfn != "" {
		predfile, err = os.Create(*predfn)
		if err != nil {
			log.Fatal(err)
		}
		defer predfile.Close()
	}

	var bb CloudForest.VoteTallyer
	switch {
	case sum:
		bb = CloudForest.NewSumBallotBox(data.Data[0].Length())

	case !cat && (num || strings.HasPrefix(forest.Target, "N")):
		bb = CloudForest.NewNumBallotBox(data.Data[0].Length())

	default:
		bb = CloudForest.NewCatBallotBox(data.Data[0].Length())
	}

	for _, tree := range forest.Trees {
		tree.Vote(data, bb)
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		fmt.Printf("Target is %v in feature %v\n", forest.Target, targeti)
		er := bb.TallyError(data.Data[targeti])
		fmt.Printf("Error: %v\n", er)
	}
	if *predfn != "" {
		fmt.Printf("Outputting label predicted actual tsv to %v\n", *predfn)
		for i, l := range data.CaseLabels {
			actual := "NA"
			if hasTarget {
				actual = data.Data[targeti].GetStr(i)
			}

			result := ""

			if sum || forest.Intercept != 0.0 {
				numresult := 0.0
				if sum {
					numresult = bb.(*CloudForest.SumBallotBox).TallyNum(i) + forest.Intercept
				} else {
					numresult = bb.(*CloudForest.NumBallotBox).TallyNum(i) + forest.Intercept
				}
				if expit {
					numresult = CloudForest.Expit(numresult)
				}
				result = fmt.Sprintf("%v", numresult)

			} else {
				result = bb.Tally(i)
			}
			fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, result, actual)
		}
	}

	//Not thread safe code!
	if *votefn != "" {
		fmt.Printf("Outputting vote totals to %v\n", *votefn)
		cbb := bb.(*CloudForest.CatBallotBox)
		votefile, err := os.Create(*votefn)
		if err != nil {
			log.Fatal(err)
		}
		defer votefile.Close()
		fmt.Fprintf(votefile, ".")

		for _, lable := range cbb.CatMap.Back {
			fmt.Fprintf(votefile, "\t%v", lable)
		}
		fmt.Fprintf(votefile, "\n")

		for i, box := range cbb.Box {
			fmt.Fprintf(votefile, "%v", data.CaseLabels[i])

			for j, _ := range cbb.CatMap.Back {
				total := 0.0
				total = box.Map[j]

				fmt.Fprintf(votefile, "\t%v", total)

			}
			fmt.Fprintf(votefile, "\n")

		}
	}
}
