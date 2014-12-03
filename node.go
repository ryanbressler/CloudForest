package CloudForest

import ()

//Recursable defines a function signature for functions that can be called at every
//down stream node of a tree as Node.Recurse recurses up the tree. The function should
//have two parameters, the current node and an array of ints specifying the cases that
//have not been split away.
type Recursable func(*Node, []int, int)

type CodedRecursable func(*Node, *[]int, int, int) (int, interface{}, int)

//A node of a decision tree.
//Pred is a string containing either the category or a representation of a float
//(less then ideal)
type Node struct {
	CodedSplit interface{}
	Featurei   int
	Left       *Node
	Right      *Node
	Missing    *Node
	Pred       string
	Splitter   *Splitter
	Members    []int
}

//vist each child node with the supplied function
func (n *Node) Climb(c func(*Node)) {
	c(n)
	if n.Left != nil {
		n.Left.Climb(c)
	}
	if n.Right != nil {
		n.Right.Climb(c)
	}
	if n.Missing != nil {
		n.Missing.Climb(c)
	}
}

//Recurse is used to apply a Recursable function at every downstream node as the cases
//specified by case []int are split using the data in fm *Featurematrix. Recursion
//down a branch stops when a a node with n.Splitter == nil is reached. Recursion down
//the Missing branch is only used if n.Missing!=nil.
//For example votes can be tabulated using code like:
//	t.Root.Recurse(func(n *Node, cases []int) {
//		if n.Left == nil && n.Right == nil {
//			// I'm in a leaf node
//			for i := 0; i < len(cases); i++ {
//				bb.Vote(cases[i], n.Pred)
//			}
//		}
//	}, fm, cases)
func (n *Node) Recurse(r Recursable, fm *FeatureMatrix, cases []int, depth int) {
	r(n, cases, depth)
	depth++
	var ls, rs, ms []int
	switch {
	case n.CodedSplit != nil:
		ls, rs, ms = fm.Data[n.Featurei].Split(n.CodedSplit, cases)
	case n.Splitter != nil:
		ls, rs, ms = n.Splitter.Split(fm, cases)
	default:
		return
	}
	n.Left.Recurse(r, fm, ls, depth)
	n.Right.Recurse(r, fm, rs, depth)
	if len(ms) > 0 && n.Missing != nil {
		n.Missing.Recurse(r, fm, ms, depth)
	}
}

func (n *Node) CodedRecurse(r CodedRecursable, fm *FeatureMatrix, cases *[]int, depth int, nconstantsbefore int) {
	fi, codedSplit, nconstants := r(n, cases, depth, nconstantsbefore)
	depth++
	if codedSplit != nil {
		li, ri := fm.Data[fi].SplitPoints(codedSplit, cases)
		cs := (*cases)[:li]
		n.Left.CodedRecurse(r, fm, &cs, depth, nconstants)
		cs = (*cases)[ri:]
		n.Right.CodedRecurse(r, fm, &cs, depth, nconstants)
		if li != ri && n.Missing != nil {
			cs = (*cases)[li:ri]
			n.Missing.CodedRecurse(r, fm, &cs, depth, nconstants)
		}
	}
}
