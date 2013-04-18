package CloudForest

import ()

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
			}

		}
	}

}

//BUG(ryan) not done yet ... just wires together some stubs like BestSplitte
//Grow the tree as a random forest tree
func (t *Tree) Grow(fm *FeatureMatrix, target *Feature, cases []int, mTry int, leafSize int) {
	t.Root.Recurse(func(n *Node, cases []int) {
		if leafSize < len(cases) {
			best := BestSplitter(fm, target, cases, mTry)
			//TODO: see if split is good enough
			n.Splitter = best
			n.Left = new(Node)
			n.Right = new(Node)
			goto EndRecurse
		}
		//This is a leaf node so we need to find the predictive value

		switch target.Numerical {
		case true:
			bb := NewBallotBox(len(cases))
			for i := range cases {
				if !target.Missing[i] {
					bb.VoteNum(i, target.Data[i])
				}

			}

		case false:
			bb := NewBallotBox(len(cases))
			for i := range cases {
				if !target.Missing[i] {
					bb.VoteCat(i, target.Back[target.Data[i]])
				}

			}

		}

	EndRecurse:
	}, fm, cases)
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

//GetLeaves is called by the leaf count utility to
//gather statistics about the nodes of a tree including the sets of cases at
//"leaf" nodes that aren't split further and the number of times each feature
//is used to split away each case.
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

//Leaf is a struct for storing the index of the cases at a terminal "Leaf" node
//along with the Numeric predicted value.
type Leaf struct {
	Cases []int
	Pred  string
}

//Tree.Vote casts a vote for the predicted value of each case in fm *FeatureMatrix.
//into bb *BallotBox. Since BallotBox is not thread safe trees should not vote
//into the same BallotBox in parralel. The target feature should be provided but need not
//have data ...only target.Numerical is used currentelly.
func (t *Tree) Vote(fm *FeatureMatrix, bb *BallotBox, target *Feature) {
	ncases := len(fm.Data[0].Data)
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int) {
		if n.Left == nil && n.Right == nil {
			// I'm in a leaf node
			for i := 0; i < len(cases); i++ {
				bb.Vote(cases[i], n.Pred, target.Numerical)
			}
		}
	}, fm, cases)
}
