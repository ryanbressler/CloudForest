package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

type Num float64

type Forest struct {
	/*Forest string
	Target string
	Ntrees int
	Categories string
	Shrinkage int*/
	Trees []*Tree
}

type Tree struct {
	//Tree int
	Root *Node
}

func (t *Tree) AddNode(path string, pred Num, splitter *Splitter) {
	n := new(Node)
	n.Pred = pred
	n.Splitter = splitter
	if t.Root == nil {
		t.Root = n
	} else {
		loc := t.Root
		for i := 0; i < len(path); i++ {
			switch path[i : i+1] {
			case "L":
				if loc.Left == nil {
					loc.Left = n
				}
				loc = loc.Left
			case "R":
				if loc.Right == nil {
					loc.Right = n
				}
				loc = loc.Right
			}

		}
	}

}

type Node struct {
	Left     *Node
	Right    *Node
	Pred     Num
	Splitter *Splitter
}

type Splitter struct {
	Feature   string
	Numerical bool
	Value     Num
	Left      map[string]bool
	Right     map[string]bool
}

func ParseRfAcePredictor(input io.Reader) *Forest {
	r := bufio.NewReader(input)
	var forest *Forest
	var tree *Tree
	for {
		line, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}
		parsed := ParseRfAcePredictorLine(line)
		switch {
		case strings.HasPrefix(line, "FOREST"):
			forest = new(Forest)

		case strings.HasPrefix(line, "TREE"):
			tree = new(Tree)
			forest.Trees = append(forest.Trees, tree)

		case strings.HasPrefix(line, "NODE"):
			var splitter *Splitter

			pred, err := strconv.ParseFloat(parsed["PRED"], 64)
			if err != nil {
				log.Print("Error parsing predictor value ", err)
			}

			if stype, ok := parsed["SPLITTERTYPE"]; ok {
				splitter = new(Splitter)
				splitter.Feature = parsed["SPLITTER"]
				switch stype {
				case "CATEGORICAL":
					splitter.Numerical = false

					splitter.Left = make(map[string]bool)
					for _, f := range strings.Split(parsed["LVALUES"], ",") {
						splitter.Left[f] = true
					}

					splitter.Right = make(map[string]bool)
					for _, f := range strings.Split(parsed["RVALUES"], ",") {
						splitter.Right[f] = true
					}

				case "NUMERICAL":
					splitter.Numerical = true
					lvalue, err := strconv.ParseFloat(parsed["LVALUES"], 64)
					if err != nil {
						log.Print("Error parsing lvalues value ", err)
					}
					splitter.Value = Num(lvalue)
				}
			}

			tree.AddNode(parsed["NODE"], Num(pred), splitter)

		}
	}

	return forest

}

func ParseRfAcePredictorLine(line string) map[string]string {
	clauses := make([]string, 0)
	insidequotes := make([]string, 0)
	terms := strings.Split(strings.TrimSpace(line), ",")
	for _, term := range terms {
		term = strings.TrimSpace(term)
		quotes := strings.Count(term, "\"")
		if quotes == 1 || len(insidequotes) > 0 {
			insidequotes = append(insidequotes, term)
		} else {
			clauses = append(clauses, term)
		}
		if quotes == 1 && len(insidequotes) > 1 {
			clauses = append(clauses, strings.Join(insidequotes, ","))
			insidequotes = make([]string, 0)
		}
	}
	parsed := make(map[string]string, 0)
	for _, clause := range clauses {
		vs := strings.Split(clause, "=")
		for i, v := range vs {
			vs[i] = strings.Trim(strings.TrimSpace(v), "\"")
		}
		parsed[vs[0]] = vs[1]
	}

	return parsed
}

//this data strucutre is related to the one in rf-ace. In the future we may wish to represent
//catagorical features usings ints to speed up hash look up etc.
type Feature struct {
	Data      []Num
	Missing   []bool
	Numerical bool
	Map       map[string]Num
	Back      map[Num]string
	Name      string
}

func NewFeature(record []string, capacity int) Feature {
	f := Feature{make([]Num, 0, capacity), make([]bool, 0, capacity), false, make(map[string]Num, capacity), make(map[Num]string, capacity), record[0]}
	switch record[0][0:2] {
	case "N:":
		//Numerical
		f.Numerical = true
		for i := 1; i < len(record); i++ {
			v, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				f.Data = append(f.Data, Num(0))
				f.Missing = append(f.Missing, true)
				continue
			}
			f.Data = append(f.Data, Num(v))
			f.Missing = append(f.Missing, false)

		}

	default:
		//Assume Catagorical
		f.Numerical = false
		fvalue := Num(0.0)
		for i := 1; i < len(record); i++ {
			v := record[i]
			norm := strings.ToLower(v)
			if norm == "?" || norm == "nan" || norm == "na" || norm == "null" {
				f.Data = append(f.Data, Num(0))
				f.Missing = append(f.Missing, true)
				continue
			}
			nv, exsists := f.Map[v]
			if exsists == false {
				f.Map[v] = fvalue
				f.Back[fvalue] = v
				nv = fvalue
				fvalue += 1.0
			}
			f.Data = append(f.Data, Num(nv))
			f.Missing = append(f.Missing, false)

		}

	}
	return f

}

func ParseAFM(input io.Reader) []Feature {
	data := make([]Feature, 0, 100)
	tsv := csv.NewReader(input)
	tsv.Comma = '\t'
	_, err := tsv.Read()
	if err == io.EOF {
		return data
	} else if err != nil {
		log.Print("Error:", err)
		return data
	}
	capacity := tsv.FieldsPerRecord

	for {
		record, err := tsv.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Print("Error:", err)
			break
		}
		data = append(data, NewFeature(record, capacity))
	}
	return data
}

func main() {
	fm := flag.String("fm", "featurematrix.afm", "AFM formated feature matrix to use.")
	rf := flag.String("rfpred", "rface.sf", "A predictor forest as outputed by rf-ace")
	flag.Parse()

	datafile, err := os.Open(*fm) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer datafile.Close()
	data := ParseAFM(datafile)
	log.Print("Data file ", len(data), " by ", len(data[0].Data))

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forest := ParseRfAcePredictor(forestfile)
	log.Print("Fores has ", len(forest.Trees), " trees ")

}
