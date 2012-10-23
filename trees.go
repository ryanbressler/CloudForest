package main

import (
	"bufio"
	"io"
	"log"
	"strconv"
	"strings"
)

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

func (t *Tree) GetLeaves(fm *FeatureMatrix, fbycase *SparseCounter) []Leaf {
	leaves := make([]Leaf, 0)
	ncases := len(fm.Data[0].Data)
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		if n.Left == nil && n.Right == nil {
			leaves = append(leaves, Leaf{cases, n.Pred})
		}
		if fbycase != nil && n.Splitter != nil {
			for _, c := range cases {
				fbycase.Add(c, fm.Map[n.Splitter.Feature], 1)
			}
		}

	}, fm, cases)
	return leaves

}

type Leaf struct {
	Cases []int
	Pred  Num
}

type Recursable func(*Node, []int)

type Node struct {
	Left     *Node
	Right    *Node
	Pred     Num
	Splitter *Splitter
}

func (n *Node) Recurse(r Recursable, fm *FeatureMatrix, cases []int) {
	r(n, cases)
	if n.Splitter != nil {
		ls, rs := n.Splitter.Split(fm, cases)
		n.Left.Recurse(r, fm, ls)
		n.Right.Recurse(r, fm, rs)
	}
}

type Splitter struct {
	Feature   string
	Numerical bool
	Value     Num
	Left      map[string]bool
	Right     map[string]bool
}

func (s *Splitter) Split(fm *FeatureMatrix, cases []int) ([]int, []int) {
	l := make([]int, 0)
	r := make([]int, 0)
	f := fm.Data[fm.Map[s.Feature]]

	switch s.Numerical {
	case true:
		for _, i := range cases {
			if f.Missing[i] == false {
				switch {
				case f.Data[i] <= s.Value:
					l = append(l, i)
				default:
					r = append(r, i)
				}
			}

		}
	case false:
		for _, i := range cases {
			if f.Missing[i] == false {

				v := f.Back[f.Data[i]]
				switch {
				case s.Left[v]:
					l = append(l, i)
				case s.Right[v]:
					r = append(r, i)
				}
			}

		}
	}

	return l, r
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
					for _, f := range strings.Split(parsed["LVALUES"], ":") {
						splitter.Left[f] = true
					}

					splitter.Right = make(map[string]bool)
					for _, f := range strings.Split(parsed["RVALUES"], ":") {
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
