package CloudForest

import ()

//Recursable defines a function signature for functions that can be called at every
//down stream node of a tree as Node.Recurse recurses up the tree. The function should
//have two parameters, the current node and an array of ints specifying the cases that
//have not been split away.
type Recursable func(*Node, []int, int)

//A node of a decision tree.
//Pred is a string containing either the category or a representation of a float
//(less then ideal)
type Node struct {
	Left     *Node
	Right    *Node
	Missing  *Node
	Pred     string
	Splitter *Splitter
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
	if n.Splitter != nil {
		ls, rs, ms := n.Splitter.Split(fm, cases)
		n.Left.Recurse(r, fm, ls, depth)
		n.Right.Recurse(r, fm, rs, depth)
		if len(ms) > 0 && n.Missing != nil {
			n.Missing.Recurse(r, fm, ms, depth)
		}
	}
}
