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

	libsvmtarget := flag.String("libsvmtarget",
		"", "Output lib svm with the named feature in the first position.")

	anontarget := flag.String("anontarget",
		"", "Strip strings with named feature in the first position.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

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
