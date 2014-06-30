package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/ryanbressler/CloudForest"
	"io"
	"log"
	"os"
)

func openfiles(trainfn string, testfn string) (trainW io.WriteCloser, testW io.WriteCloser) {

	trainfo, err := os.Create(trainfn)
	if err != nil {
		log.Fatal(err)
	}
	testfo, err := os.Create(testfn)
	if err != nil {
		log.Fatal(err)
	}
	trainW = trainfo
	testW = testfo
	// if zipoutput {
	// 	trainz := zip.NewWriter(trainfo)
	// 	trainW, err = trainz.Create(trainfn)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	//defer trainz.Close()
	// 	testz := zip.NewWriter(testfo)
	// 	testW, err = testz.Create(testfn)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	//defer testz.Close()
	// }

	return

}

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing data.")

	blacklist := flag.String("blacklist",
		"", "A list of feature id's to exclude from the set of predictors.")

	targetname := flag.String("target",
		"", "The row header of the target in the feature matrix.")
	train := flag.String("train",
		"train_%v.fm", "Format string for training fms.")
	test := flag.String("test",
		"test_%v.fm", "Format string for testing fms.")

	// var zipoutput bool
	// flag.BoolVar(&zipoutput, "zip", false, "Output ziped files.")
	var unstratified bool
	flag.BoolVar(&unstratified, "unstratified", false, "Force unstratified sampeling of categorical target.")

	var writelibsvm bool
	flag.BoolVar(&writelibsvm, "writelibsvm", false, "Output libsvm.")

	var writearff bool
	flag.BoolVar(&writearff, "writearff", false, "Output arff.")

	var writeall bool
	flag.BoolVar(&writeall, "writeall", false, "Output all three formats.")

	var folds int
	flag.IntVar(&folds, "folds", 5, "Number of folds to generate.")

	var maxcats int
	flag.IntVar(&maxcats, "maxcats", -1, "Maximum number of categories to allow in a feature.")

	var impute bool
	flag.BoolVar(&impute, "impute", false, "Impute missing values to feature mean/mode.")

	var onehot bool
	flag.BoolVar(&onehot, "onehot", false, "Do one hot encoding of categorical features to boolean true false.")

	var num bool
	flag.BoolVar(&num, "num", false, "Do one hot encoding of categorical features to numerical features.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
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
			if id[0] == *targetname {
				continue
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

	newdata := make([]CloudForest.Feature, 0, len(data.Data)-blacklisted)
	newmap := make(map[string]int, len(data.Data)-blacklisted)

	for i, f := range data.Data {
		if !blacklistis[i] && (maxcats == -1 || f.NCats() <= maxcats) {
			newmap[f.GetName()] = len(newdata)
			newdata = append(newdata, f)
		}
	}

	data.Data = newdata
	data.Map = newmap

	if impute {
		fmt.Println("Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	if onehot {
		fmt.Println("OneHot encoding.")
		data.OneHot()
	}

	if num {
		fmt.Println("Numerical OneHot encoding.")
		data.EncodeToNum()
	}

	foldis := make([][]int, 0, folds)

	foldsize := len(data.CaseLabels) / folds
	fmt.Printf("%v cases, foldsize %v\n", len(data.CaseLabels), foldsize)
	for i := 0; i < folds; i++ {
		foldis = append(foldis, make([]int, 0, foldsize))
	}

	var targetf CloudForest.Feature

	//find the target feature
	fmt.Printf("Target : %v\n", *targetname)
	targeti, ok := data.Map[*targetname]
	if !ok {
		fmt.Println("Target not found in data, doing unstratified sampeling.")
		unstratified = true
	}

	if ok {
		targetf = data.Data[targeti]

		switch targetf.(type) {
		case *CloudForest.DenseNumFeature:
			unstratified = true
		}
	}
	if unstratified {
		ncases := len(data.CaseLabels)
		cases := make([]int, ncases, ncases)
		for i := 0; i < ncases; i++ {
			cases[i] = i
		}
		CloudForest.SampleFirstN(&cases, nil, len(cases), 0)
		for j := 0; j < folds; j++ {
			for k := j * foldsize; k < (j+1)*foldsize; k++ {
				foldis[j] = append(foldis[j], cases[k])
			}
		}

	} else {
		//sample folds stratified by case
		fmt.Printf("Stratifying by %v classes.\n", targetf.(*CloudForest.DenseCatFeature).NCats())
		bSampler := CloudForest.NewBalancedSampler(targetf.(*CloudForest.DenseCatFeature))

		fmt.Printf("Stratifying by %v classes.\n", len(bSampler.Cases))
		var samples []int
		for i := 0; i < len(bSampler.Cases); i++ {
			fmt.Printf("%v cases in class %v.\n", len(bSampler.Cases[i]), i)
			//shuffle in place
			CloudForest.SampleFirstN(&bSampler.Cases[i], &samples, len(bSampler.Cases[i]), 0)
			stratFoldSize := len(bSampler.Cases[i]) / folds
			for j := 0; j < folds; j++ {
				for k := j * stratFoldSize; k < (j+1)*stratFoldSize; k++ {
					foldis[j] = append(foldis[j], bSampler.Cases[i][k])

				}
			}

		}
	}

	trainis := make([]int, 0, foldsize*(folds-1))
	//Write training and testing matrixes
	for i := 0; i < folds; i++ {

		trainfn := fmt.Sprintf(*train, i)
		testfn := fmt.Sprintf(*test, i)

		trainis = trainis[0:0]
		for j := 0; j < folds; j++ {
			if i != j {
				trainis = append(trainis, foldis[j]...)
			}
		}

		if writearff || writeall {
			trainW, testW := openfiles(trainfn+".arff", testfn+".arff")
			CloudForest.WriteArffCases(data, foldis[i], *targetname, testW)
			CloudForest.WriteArffCases(data, trainis, *targetname, trainW)
		}

		if ((!writelibsvm) && (!writearff)) || writeall {
			trainW, testW := openfiles(trainfn, testfn)
			data.WriteCases(testW, foldis[i])
			data.WriteCases(trainW, trainis)
		}

		if writelibsvm || writeall {
			trainW, testW := openfiles(trainfn+".libsvm", testfn+".libsvm")
			CloudForest.WriteLibSvmCases(data, foldis[i], *targetname, testW)
			CloudForest.WriteLibSvmCases(data, trainis, *targetname, trainW)
		}

		fmt.Printf("Wrote fold %v. %v testing cases and %v training cases.\n", i, len(foldis[i]), len(trainis))
	}

}
