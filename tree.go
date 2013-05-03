package CloudForest

import ()

//Tree represents a single decision tree.
type Tree struct {
	//Tree int
	Root   *Node
	Target string
}

func NewTree() *Tree {
	return &Tree{new(Node), ""}
}

//AddNode adds a node a the specified path with the specivied pred value and/or
//Splitter. Paths are specified in the same format as in rf-aces sf files, as a
//string of 'L' and 'R'. Nodes must be added from the root up as the case where
//the path specifies a node whose parent does not allready exist in the tree is
//not handled well.
func (t *Tree) AddNode(path string, pred string, splitter *Splitter) {
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

			case "M":
				if loc.Missing == nil {
					loc.Missing = n
				}
				loc = loc.Missing
			}

		}
	}

}

/*
tree.Grow grows the reciever tree through recursion. It uses impurity decrease to selct spliters at
each node as in Brieman's Random Forest. It should be called on a tree with only a root node defined.

fm is a feature matrix of training data.

target is the feature to predict via regression or classification as determined by feature type.

cases specifies the cases to calculate impurity decrease over and can contain repeated values
to allow for sampeling of cases with replacment as in RF.

mTry specifies the number of canidate features to evaluate for each split.

leafSize specifies the minimum number of cases at a leafNode.
*/
func (t *Tree) Grow(fm *FeatureMatrix,
	target Target,
	cases []int,
	canidates []int,
	mTry int,
	leafSize int,
	itter bool,
	splitmissing bool,
	importance *[]RunningMean,
	l *[]int,
	r *[]int,
	m *[]int) {

	var rm []RunningMean
	if importance != nil {
		rm = *importance
	}
	t.Root.Recurse(func(n *Node, innercases []int) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&canidates, mTry)
			best, impDec := fm.BestSplitter(target, innercases, canidates[:mTry], itter, l, r, m)
			if best != nil && impDec > minImp {
				if importance != nil {
					rm[fm.Map[best.Feature]].Add(impDec)
				}
				//not a leaf node so define the spliter and left and right nodes
				//so recursion will continue
				n.Splitter = best
				n.Pred = ""
				n.Left = new(Node)
				n.Right = new(Node)
				if splitmissing {
					n.Missing = new(Node)
				}
				return
			}
		}

		//Leaf node so find the predictive value and set it in n.Pred
		n.Splitter = nil
		n.Pred = target.FindPredicted(innercases)

	}, fm, cases)
}

//GetSplits returns the arrays of all Numeric spliters of a tree.
func (t *Tree) GetSplits(fm *FeatureMatrix, fbycase *SparseCounter, relativeSplitCount *SparseCounter) []Splitter {
	splitters := make([]Splitter, 0)
	ncases := len(fm.Data[0].Missing) // grab the number of samples for the first feature
	cases := make([]int, ncases)      //make an array that large
	for i, _ := range cases {
		cases[i] = i
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		//if we're on a splitting node
		if fbycase != nil && n.Splitter != nil && n.Splitter.Numerical == true {
			//add this splitter to the list
			splitters = append(splitters, Splitter{n.Splitter.Feature,
				n.Splitter.Numerical,
				n.Splitter.Value,
				n.Splitter.Left})
			f_id := n.Splitter.Feature               //get the feature at this splitter
			f := fm.Data[fm.Map[n.Splitter.Feature]] //get the feature at this splitter
			for _, c := range cases {                //for each case

				if f.Missing[c] == false { //if there isa value for this case
					fbycase.Add(c, fm.Map[f_id], 1) //count the number of times each case is present for a split by a feature

					switch f.NumData[c] <= n.Splitter.Value {
					case true: //if the value was split to the left
						relativeSplitCount.Add(c, fm.Map[f_id], -1) //subtract one
					case false:
						relativeSplitCount.Add(c, fm.Map[f_id], 1) //add one
					}
				}
			}
		}
	}, fm, cases)
	return splitters //return the array of all splitters from the tree

}

//GetLeaves is called by the leaf count utility to
//gather statistics about the nodes of a tree including the sets of cases at
//"leaf" nodes that aren't split further and the number of times each feature
//is used to split away each case.
func (t *Tree) GetLeaves(fm *FeatureMatrix, fbycase *SparseCounter) []Leaf {
	leaves := make([]Leaf, 0)
	ncases := len(fm.Data[0].Missing)
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

//Leaf is a struct for storing the index of the cases at a terminal "Leaf" node
//along with the Numeric predicted value.
type Leaf struct {
	Cases []int
	Pred  string
}

//Tree.Vote casts a vote for the predicted value of each case in fm *FeatureMatrix.
//into bb *BallotBox. Since BallotBox is not thread safe trees should not vote
//into the same BallotBox in parralel.
func (t *Tree) Vote(fm *FeatureMatrix, bb VoteTallyer) {
	ncases := len(fm.Data[0].Missing)
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
