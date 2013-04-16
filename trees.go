package CloudForest

import (
	"bufio"
	"io"
	"log"
	"strconv"
	"strings"
)

//Forest represents a collection of decision trees grown to predict Target.
type Forest struct {
	//Forest string
	Target string
	//Ntrees int
	//Categories string
	//Shrinkage int*/
	Trees []*Tree
}

//Tree represents a single decision tree.
type Tree struct {
	//Tree int
	Root *Node
}

//AddNode adds a node a the specified path with the specivied pred value and/or
//Splitter. Paths are specified in the same format as in rf-aces sf files, as a 
//string of 'L' and 'R'. Nodes must be added from the root up as the case where 
//the path specifies a node whose parent does not allready exist in the tree is 
//not handled well.
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

//Returns the arrays of all spliters of a tree.
func (t *Tree) GetSplits(fm *FeatureMatrix, fbycase *SparseCounter, relativeSplitCount *SparseCounter) []Splitter {
	splitters := make([]Splitter, 0)
	ncases := len(fm.Data[0].Data) // grab the number of samples for the first feature
	cases := make([]int, ncases)   //make an array that large
	for i, _ := range cases {
		cases[i] = i
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		//if we're on a splitting node
		if fbycase != nil && n.Splitter != nil {
			//add this splitter to the list
			splitters = append(splitters, Splitter{n.Splitter.Feature, n.Splitter.Numerical, n.Splitter.Value, n.Splitter.Left, n.Splitter.Right})
			f_id := n.Splitter.Feature               //get the feature at this splitter
			f := fm.Data[fm.Map[n.Splitter.Feature]] //get the feature at this splitter
			for _, c := range cases {                //for each case

				if f.Missing[c] == false { //if there isa value for this case
					fbycase.Add(c, fm.Map[f_id], 1) //count the number of times each case is present for a split by a feature
					fvalue := f.Back[f.Data[c]]     //what is the feature value for this case?

					switch {
					case n.Splitter.Left[fvalue]: //if the value was split to the left
						relativeSplitCount.Add(c, fm.Map[f_id], -1) //subtract one
					case n.Splitter.Right[fvalue]:
						relativeSplitCount.Add(c, fm.Map[f_id], 1) //add one
					}
				}
			}
		}
	}, fm, cases)
	return splitters //return the array of all splitters from the tree

}

//Gather the leaves of a tree.
func (t *Tree) GetLeaves(fm *FeatureMatrix, fbycase *SparseCounter) []Leaf {
	leaves := make([]Leaf, 0)
	ncases := len(fm.Data[0].Data)
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		if n.Left == nil && n.Right == nil { // I'm in a leaf node
			leaves = append(leaves, Leaf{cases, n.Pred})
		}
		if fbycase != nil && n.Splitter != nil { //I'm not in a leaf node?
			for _, c := range cases {
				fbycase.Add(c, fm.Map[n.Splitter.Feature], 1)
			}
		}

	}, fm, cases)
	return leaves

}

//Tree.Vote casts a vote for the predicted value of each case in fm *FeatureMatrix.
//into bb *BallotBox. Since BallotBox is not thread safe trees should not vote
//into the same BallotBox in parralel. 
func (t *Tree) Vote(fm *FeatureMatrix, bb *BallotBox) {
	ncases := len(fm.Data)
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		if n.Left == nil && n.Right == nil {
			// I'm in a leaf node
			for i := 0; i < len(cases); i++ {
				bb.Vote(cases[i], n.Pred)
			}
		}
	}, fm, cases)
}

type Leaf struct {
	Cases []int
	Pred  Num
}

//Recurssable defines a function signature for functions that can be called at every
//down stream node of a tree as Node.Recurse recurses up the tree. The function should
//have two paramaters, the current node and an array of ints specifying the cases that
//have not been split away.
type Recursable func(*Node, []int)

//A node of a decision tree.
type Node struct {
	Left     *Node
	Right    *Node
	Pred     Num
	Splitter *Splitter
}

//Recurse is used to apply a Recursable function at every downstream node as the cases
//specified by case []int are split using the data in fm *Featurematrix.
//For example votes can be tabulated using code like
//	t.Root.Recurse(func(n *Node, cases []int) {
//		if n.Left == nil && n.Right == nil {
//			// I'm in a leaf node
//			for i := 0; i < len(cases); i++ {
//				bb.Vote(cases[i], n.Pred)
//			}
//		}
//	}, fm, cases)
func (n *Node) Recurse(r Recursable, fm *FeatureMatrix, cases []int) {
	r(n, cases)
	if n.Splitter != nil {
		ls, rs := n.Splitter.Split(fm, cases)
		n.Left.Recurse(r, fm, ls)
		n.Right.Recurse(r, fm, rs)
	}
}

//Splitter contains fields that can be used to cases by a single feature. The split
//can be either numerical in which case it is defined by the Value field or 
//catagorical in which case it is defined by the Left and Right fields.
type Splitter struct {
	Feature   string
	Numerical bool
	Value     Num
	Left      map[string]bool
	Right     map[string]bool
}

//Splitter.Split seperates cases []int using the data in fm *FeatureMatrix
//and returns left and right []ints.
//It applies either a Numerical or Catagorical split. In the Numerical case
//everything <= to Value is sent left; for the Catagorical case a look up 
//table is used.
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

//ParseRfAcePredictor reads a forest from an io.Reader.
//The forest should be in rf-ace's "stoicastic forest" sf format
//It ignores fields that are not use by cloud forest.
// Start of an example file:
// FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800,CATEGORIES="",SHRINKAGE=0
// TREE=0
// NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false",RVALUES="true"
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
			forest.Target = parsed["TARGET"]

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

//ParseRfAcePredictorLine parses a single line of an rf-ace sf "stoicastic forest"
//and returns a map[string]string of the key value pairs
//Some examples of valid input lines:
// FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800,CATEGORIES="",SHRINKAGE=0
// TREE=0
// NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false",RVALUES="true"
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
