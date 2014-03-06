package CloudForest

import ()

//Tree represents a single decision tree.
type Tree struct {
	//Tree int
	Root   *Node
	Target string
	Weight float64
}

func NewTree() *Tree {
	return &Tree{new(Node), "", -1.0}
}

//AddNode adds a node a the specified path with the specified pred value and/or
//Splitter. Paths are specified in the same format as in rf-aces sf files, as a
//string of 'L' and 'R'. Nodes must be added from the root up as the case where
//the path specifies a node whose parent does not already exist in the tree is
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
tree.Grow grows the receiver tree through recursion. It uses impurity decrease to select splitters at
each node as in Brieman's Random Forest. It should be called on a tree with only a root node defined.

fm is a feature matrix of training data.

target is the feature to predict via regression or classification as determined by feature type.

cases specifies the cases to calculate impurity decrease over and can contain repeated values
to allow for sampling of cases with replacement as in RF.

canidates specifies the potential features to use as splitters

mTry specifies the number of candidate features to evaluate for each split.

leafSize specifies the minimum number of cases at a leafNode.

splitmissing indicates if missing values should be split onto a third branch

vet indicates if splits should be penalized against a randomized version of them selves
*/
func (t *Tree) Grow(fm *FeatureMatrix,
	target Target,
	cases []int,
	candidates []int,
	oob []int,
	mTry int,
	leafSize int,
	splitmissing bool,
	vet bool,
	evaloob bool,
	importance *[]*RunningMean,
	depthUsed *[]int,
	allocs *BestSplitAllocs) {

	var innercanidates []int
	t.Root.CodedRecurse(func(n *Node, innercases []int, depth int) (fi int, split interface{}) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&candidates, mTry)
			innercanidates = candidates[:mTry]
			var impDec float64
			fi, split, impDec = fm.BestSplitter(target, &innercases, &innercanidates, &oob, leafSize, vet, evaloob, allocs)
			if impDec > minImp {
				if importance != nil {
					(*importance)[fi].Add(impDec)
				}
				if depthUsed != nil && ((*depthUsed)[fi] == 0 || depth < (*depthUsed)[fi]) {
					(*depthUsed)[fi] = depth
				}
				//not a leaf node so define the splitter and left and right nodes
				//so recursion will continue
				n.Splitter = fm.Data[fi].DecodeSplit(split)
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
		split = nil
		n.Splitter = nil
		n.Pred = target.FindPredicted(innercases)
		return

	}, fm, cases, 0)
}

//GetLeaves is called by the leaf count utility to
//gather statistics about the nodes of a tree including the sets of cases at
//"leaf" nodes that aren't split further and the number of times each feature
//is used to split away each case.
func (t *Tree) GetLeaves(fm *FeatureMatrix, fbycase *SparseCounter) []Leaf {
	leaves := make([]Leaf, 0)
	ncases := fm.Data[0].Length()
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int, depth int) {
		if n.Left == nil && n.Right == nil { // I'm in a leaf node
			leaves = append(leaves, Leaf{cases, n.Pred})
		}
		if fbycase != nil && n.Splitter != nil { //I'm not in a leaf node?
			for _, c := range cases {
				fbycase.Add(c, fm.Map[n.Splitter.Feature], 1)
			}
		}

	}, fm, cases, 0)
	return leaves

}

func (t *Tree) Partition(fm *FeatureMatrix) *[][]int {
	leaves := make([][]int, 0)
	ncases := fm.Data[0].Length()
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int, depth int) {
		if n.Left == nil && n.Right == nil { // I'm in a leaf node
			leaves = append(leaves, cases)
		}

	}, fm, cases, 0)
	return &leaves

}

//Leaf is a struct for storing the index of the cases at a terminal "Leaf" node
//along with the Numeric predicted value.
type Leaf struct {
	Cases []int
	Pred  string
}

//Tree.Vote casts a vote for the predicted value of each case in fm *FeatureMatrix.
//into bb *BallotBox. Since BallotBox is not thread safe trees should not vote
//into the same BallotBox in parallel.
func (t *Tree) Vote(fm *FeatureMatrix, bb VoteTallyer) {
	ncases := fm.Data[0].Length()
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.VoteCases(fm, bb, cases)
}

//Tree.VoteCases casts a vote for the predicted value of each case in fm *FeatureMatrix.
//into bb *BallotBox. Since BallotBox is not thread safe trees should not vote
//into the same BallotBox in parallel.
func (t *Tree) VoteCases(fm *FeatureMatrix, bb VoteTallyer, cases []int) {

	weight := 1.0
	if t.Weight >= 0.0 {
		weight = t.Weight
	}

	t.Root.Recurse(func(n *Node, cases []int, depth int) {
		if n.Left == nil && n.Right == nil {
			// I'm in a leaf node
			for i := 0; i < len(cases); i++ {
				bb.Vote(cases[i], n.Pred, weight)
			}
		}
	}, fm, cases, 0)
}
