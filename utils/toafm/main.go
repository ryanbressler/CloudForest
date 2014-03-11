package main

import (
	"flag"
	"github.com/ryanbressler/CloudForest"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("data",
		"", "Data file to read.")

	outfn := flag.String("out",
		"", "The name of a file to write feature matrix too.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	//anotate with type information
	for _, f := range data.Data {
		switch f.(type) {
		case *CloudForest.DenseNumFeature:
			nf := f.(*CloudForest.DenseNumFeature)
			if !strings.HasPrefix(nf.Name, "N:") {
				nf.Name = "N:" + nf.Name
			}
		case *CloudForest.DenseCatFeature:
			nf := f.(*CloudForest.DenseCatFeature)
			if !(strings.HasPrefix(nf.Name, "C:") || strings.HasPrefix(nf.Name, "B:")) {
				nf.Name = "C:" + nf.Name
			}

		}
	}

	ncases := data.Data[0].Length()
	cases := make([]int, ncases, ncases)

	for i := 0; i < ncases; i++ {
		cases[i] = i
	}

	outfile, err := os.Create(*outfn)
	if err != nil {
		log.Fatal(err)
	}
	defer outfile.Close()

	err = data.WriteCases(outfile, cases)
	if err != nil {
		log.Fatal(err)
	}

}
