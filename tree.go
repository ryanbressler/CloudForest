package CloudForest

import ()

type nodeAndCases struct {
	n          *Node
	start      int
	end        int
	nconstants int
	parent     *Node
	dead       bool
}

//Tree represents a single decision tree.
type Tree struct {
	//Tree int
	Root   *Node
	Target string
	Weight float64
}

//NewTree initializes one node tree.
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

//StripCodes removes all of the coded splits from a tree so that it can be used on new catagorical data.
func (t *Tree) StripCodes() {
	t.Root.Climb(func(n *Node) {
		if n.CodedSplit != nil {
			n.CodedSplit = nil

		}
	})
}

/*
Grow grows the receiver tree through recursion. It uses impurity decrease to select splitters at
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
	maxDepth int,
	splitmissing bool,
	force bool,
	vet bool,
	evaloob bool,
	extraRandom bool,
	importance *[]*RunningMean,
	depthUsed *[]int,
	allocs *BestSplitAllocs) {

	//var innercanidates []int
	var impDec float64
	// for i := 0; i < len(allocs.Weights); i++ {
	// 	allocs.Weights[i] = 0
	// }
	// allocs.Cases = allocs.Cases[0:0]
	// for _, i := range cases {
	// 	if allocs.Weights[i] == 0 {
	// 		allocs.Cases = append(allocs.Cases, i)
	// 	}
	// 	allocs.Weights[i]++
	// }
	t.Root.CodedRecurse(func(n *Node, innercases *[]int, depth int, nconstantsbefore int) (fi int, split interface{}, nconstants int) {

		nconstants = nconstantsbefore

		if (depth < maxDepth || maxDepth <= 0) && (2*leafSize) <= len(*innercases) {
			//SampleFirstN(&candidates, &innercanidates, mTry, 0)
			//innercanidates = candidates[:mTry]

			fi, split, impDec, nconstants = fm.BestSplitter(target, innercases, &candidates, mTry, &oob, leafSize, force, vet, evaloob, extraRandom, allocs, nconstantsbefore)

			// for i := mTry; i < len(candidates)-1 && impDec == minImp; i++ {
			// 	randi := i + rand.Intn(len(candidates)-i)
			// 	candidates[randi], candidates[i] = candidates[i], candidates[randi]
			// 	innercanidates = candidates[i : i+1]
			// 	fi, split, impDec, nconstants = fm.BestSplitter(target, innercases, &innercanidates, &oob, leafSize, vet, evaloob, allocs, nconstantsbefore)
			// }
			if split != nil {
				if importance != nil {
					(*importance)[fi].Add(impDec)
				}
				if depthUsed != nil && ((*depthUsed)[fi] == 0 || depth < (*depthUsed)[fi]) {
					(*depthUsed)[fi] = depth
				}
				//not a leaf node so define the splitter and left and right nodes
				//so recursion will continue
				n.CodedSplit = split
				n.Featurei = fi
				n.Splitter = fm.Data[fi].DecodeSplit(split)
				n.Pred = ""
				//is this check needed? is it safe to reuse?
				if n.Left == nil || n.Right == nil {
					n.Left = new(Node)
					n.Right = new(Node)
				}
				if splitmissing {
					n.Missing = new(Node)
				}
				return
			}

		}
		//fmt.Println("Terminating in tree grow.")

		//Leaf node so find the predictive value and set it in n.Pred
		split = nil
		n.CodedSplit = nil
		n.Splitter = nil
		n.Pred = target.FindPredicted(*innercases)
		return

	}, fm, &cases, 0, 0)
}

func (t *Tree) GrowJungle(fm *FeatureMatrix,
	target Target,
	cases []int,
	candidates []int,
	oob []int,
	mTry int,
	leafSize int,
	maxDepth int,
	splitmissing bool,
	force bool,
	vet bool,
	evaloob bool,
	extraRandom bool,
	importance *[]*RunningMean,
	depthUsed *[]int,
	allocs *BestSplitAllocs) {

	//var innercanidates []int
	var impDec float64
	nodes := []nodeAndCases{nodeAndCases{t.Root, 0, len(cases), 0, nil, false}}
	var depth, nconstants, start, end, fi, firstThisLevel int
	var split interface{}

	innercases := cases[0:len(cases)]
	innercases2 := cases[0:len(cases)]

	for {
		//Extend Nodes
		lastThisLevel := len(nodes)

		for i := firstThisLevel; i < lastThisLevel; i++ {
			if nodes[i].dead {
				continue
			}

			node := nodes[i]
			n := node.n
			start = node.start
			end = node.end
			innercases = cases[node.start:node.end]
			nconstants = node.nconstants

			if (depth < maxDepth || maxDepth <= 0) && (2*leafSize) <= len(innercases) {

				fi, split, impDec, nconstants = fm.BestSplitter(target, &innercases, &candidates, mTry, &oob, leafSize, force, vet, evaloob, extraRandom, allocs, nconstants)

				if split != nil {
					if importance != nil {
						(*importance)[fi].Add(impDec)
					}
					if depthUsed != nil && ((*depthUsed)[fi] == 0 || depth < (*depthUsed)[fi]) {
						(*depthUsed)[fi] = depth
					}

					//not a leaf node so define the splitter and left and right nodes
					//so recursion will continue
					n.CodedSplit = split
					n.Featurei = fi
					n.Splitter = fm.Data[fi].DecodeSplit(split)
					n.Pred = ""

					li, ri := fm.Data[fi].SplitPoints(split, &innercases)

					//Left
					n.Left = new(Node)
					nodes = append(nodes, nodeAndCases{n.Left, start, start + li, nconstants, n, false})

					//Right
					n.Right = new(Node)
					nodes = append(nodes, nodeAndCases{n.Right, start + ri, end, nconstants, n, false})

					if splitmissing {
						n.Missing = new(Node)
						if li != ri {
							nodes = append(nodes, nodeAndCases{n.Missing, start + li, start + ri, nconstants, n, false})
						}
					}

					continue
				}

			}

			//Leaf node so find the predictive value and set it in n.Pred
			split = nil
			n.CodedSplit = nil
			n.Splitter = nil
			n.Pred = target.FindPredicted(innercases)
			continue

		}

		if len(nodes) == lastThisLevel {
			break
		}

		//Combine next level nodes here for jungles
		madeChanges := true
		for madeChanges {
			madeChanges = false
			for i := lastThisLevel; i < len(nodes); i++ {
				if nodes[i].dead {
					continue
				}
				var maxImpDec, impDec float64
				combine := -1
				nodei := nodes[i]
				for j := lastThisLevel; j < len(nodes); j++ {
					if nodes[j].dead {
						continue
					}

					nodej := nodes[j]
					if nodei.end == nodej.start && nodei.parent != nodej.parent {
						//should we consider other harder noncontigeous combinations?
						//or is theis enough of a random search to be fine?
						innercases = cases[nodei.start:nodei.end]
						innercases2 = cases[nodej.start:nodej.end]
						impDec = target.SplitImpurity(&innercases, &innercases2, nil, allocs)

						innercases = cases[nodei.start:nodej.end]
						impDec -= target.Impurity(&innercases, allocs.Counter)

						if impDec > maxImpDec {
							maxImpDec = impDec
							combine = j
						}

					}
				}

				if combine != -1 {
					madeChanges = true
					nodes[combine].dead = true
					nj := nodes[combine]
					nodei.end = nj.end
					parent := nj.parent
					switch nj.n {
					case parent.Left:
						parent.Left = nodei.n
					case parent.Right:
						parent.Right = nodei.n
					case parent.Missing:
						parent.Missing = nodei.n
					}

				}

			}
		}

		firstThisLevel = lastThisLevel

		depth++
	}

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

//Partition partitions all of the cases in a FeatureMatrix.
func (t *Tree) Partition(fm *FeatureMatrix) (*[][]int, *[]string) {
	leaves := make([][]int, 0)
	preds := make([]string, 0)
	ncases := fm.Data[0].Length()
	cases := make([]int, 0, ncases)
	for i := 0; i < ncases; i++ {
		cases = append(cases, i)
	}

	t.Root.Recurse(func(n *Node, cases []int, depth int) {
		if n.Left == nil && n.Right == nil { // I'm in a leaf node
			leaves = append(leaves, cases)
			preds = append(preds, n.Pred)
		}

	}, fm, cases, 0)
	return &leaves, &preds

}

//Leaf is a struct for storing the index of the cases at a terminal "Leaf" node
//along with the Numeric predicted value.
type Leaf struct {
	Cases []int
	Pred  string
}

//Vote casts a vote for the predicted value of each case in fm *FeatureMatrix.
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

//VoteCases casts a vote for the predicted value of each case in fm *FeatureMatrix.
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
