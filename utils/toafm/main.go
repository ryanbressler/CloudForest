package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/lytics/CloudForest"
	"io"
	"log"
	"os"
	"strings"
)

func main() {
	fm := flag.String("data",
		"", "Data file to read.")

	outfn := flag.String("out",
		"", "The name of a file to write feature matrix too.")

	libsvmtarget := flag.String("libsvmtarget",
		"", "Output lib svm with the named feature in the first position.")

	anontarget := flag.String("anontarget",
		"", "Strip strings with named feature in the first position.")

	blacklist := flag.String("blacklist",
		"", "A list of feature id's to exclude from the set of predictors.")

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
			if id[0] == *anontarget || id[0] == *libsvmtarget {
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
		if !blacklistis[i] {
			newmap[f.GetName()] = len(newdata)
			newdata = append(newdata, f)
		}
	}

	data.Data = newdata
	data.Map = newmap

	if *anontarget != "" {
		data.StripStrings(*anontarget)

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

	if *libsvmtarget == "" {

		err = data.WriteCases(outfile, cases)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		// targeti, ok := data.Map[*libsvmtarget]
		// if !ok {
		// 	log.Fatalf("Target '%v' not found in data.", *libsvmtarget)
		// }
		// target := data.Data[targeti]

		// data.Data = append(data.Data[:targeti], data.Data[targeti+1:]...)

		// encodedfm := data.EncodeToNum()

		// oucsv := csv.NewWriter(outfile)
		// oucsv.Comma = ' '

		// for i := 0; i < target.Length(); i++ {
		// 	entries := make([]string, 0, 10)
		// 	switch target.(type) {
		// 	case CloudForest.NumFeature:
		// 		entries = append(entries, target.GetStr(i))
		// 	case CloudForest.CatFeature:
		// 		entries = append(entries, fmt.Sprintf("%v", target.(CloudForest.CatFeature).Geti(i)))
		// 	}

		// 	for j, f := range encodedfm.Data {
		// 		v := f.(CloudForest.NumFeature).Get(i)
		// 		if v != 0.0 {
		// 			entries = append(entries, fmt.Sprintf("%v:%v", j+1, v))
		// 		}
		// 	}
		// 	//fmt.Println(entries)
		// 	err := oucsv.Write(entries)
		// 	if err != nil {
		// 		log.Fatalf("Error writing libsvm:\n%v", err)
		// 	}

		// }
		// oucsv.Flush()
		err = CloudForest.WriteLibSvm(data, *libsvmtarget, outfile)
		if err != nil {
			log.Fatalf("Error writing libsvm:\n%v", err)
		}

	}

}
