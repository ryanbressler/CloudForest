package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/lytics/CloudForest"
)

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest.")
	outf := flag.String("leaves", "leaves.tsv", "a case by case sparse matrix of leaf co-occurrence in tsv format")
	boutf := flag.String("branches", "", "a case by feature sparse matrix of leaf co-occurrence in tsv format")
	soutf := flag.String("splits", "", "a file to write a json record of splite per feature")
	var threads int
	flag.IntVar(&threads, "threads", 1, "Parse seperate forests in n seperate threads.")

	flag.Parse()

	splits := make(map[string][]string)

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	log.Print("Data file ", len(data.Data), " by ", data.Data[0].Length())

	counts := new(CloudForest.SparseCounter)
	var caseFeatureCounts *CloudForest.SparseCounter
	if *boutf != "" {
		caseFeatureCounts = new(CloudForest.SparseCounter)
	}

	files := strings.Split(*rf, ",")

	runtime.GOMAXPROCS(threads)

	fileChan := make(chan string, 0)
	doneChan := make(chan int, 0)

	go func() {
		for _, fn := range files {
			fileChan <- fn
		}
	}()

	for i := 0; i < threads; i++ {

		go func() {
			for {
				fn := <-fileChan

				forestfile, err := os.Open(fn) // For read access.
				if err != nil {
					log.Fatal(err)
				}
				defer forestfile.Close()
				forestreader := CloudForest.NewForestReader(forestfile)
				forest, err := forestreader.ReadForest()
				if err != nil {
					log.Fatal(err)
				}
				log.Print("Forest has ", len(forest.Trees), " trees ")

				for i := 0; i < len(forest.Trees); i++ {
					fmt.Print(".")
					leaves := forest.Trees[i].GetLeaves(data, caseFeatureCounts)
					for _, leaf := range leaves {
						for j := 0; j < len(leaf.Cases); j++ {
							for k := 0; k < len(leaf.Cases); k++ {

								counts.Add(leaf.Cases[j], leaf.Cases[k], 1)

							}
						}
					}

					if *soutf != "" {
						forest.Trees[i].Root.Climb(func(n *CloudForest.Node) {
							if n.Splitter != nil {
								name := n.Splitter.Feature
								_, ok := splits[name]
								if !ok {
									splits[name] = make([]string, 0, 10)
								}
								split := ""
								switch n.Splitter.Numerical {
								case true:
									split = fmt.Sprintf("%v", n.Splitter.Value)
								case false:
									keys := make([]string, 0, len(n.Splitter.Left))
									for k := range n.Splitter.Left {
										keys = append(keys, k)
									}
									split = strings.Join(keys, ",")
								}
								splits[name] = append(splits[name], split)
							}
						})
					}

				}
				doneChan <- 1
			}
		}()

	}

	for i := 0; i < len(files); i++ {
		<-doneChan
	}

	log.Print("Outputting Case Case  Co-Occurrence Counts")
	outfile, err := os.Create(*outf)
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()
	counts.WriteTsv(outfile)

	if *boutf != "" {
		log.Print("Outputting Case Feature Co-Occurrence Counts")
		boutfile, err := os.Create(*boutf)
		if err != nil {
			log.Fatal(err)
		}
		defer boutfile.Close()
		caseFeatureCounts.WriteTsv(boutfile)
	}

	if *soutf != "" {
		soutfile, err := os.Create(*soutf)
		if err != nil {
			log.Fatal(err)
		}
		defer soutfile.Close()
		encoder := json.NewEncoder(soutfile)
		encoder.Encode(splits)
	}
}
