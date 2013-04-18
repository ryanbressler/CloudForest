package CloudForest

import ()

//Recurssable defines a function signature for functions that can be called at every
//down stream node of a tree as Node.Recurse recurses up the tree. The function should
//have two paramaters, the current node and an array of ints specifying the cases that
//have not been split away.
type Recursable func(*Node, []int)

//A node of a decision tree.
type Node struct {
	Left     *Node
	Right    *Node
	Pred     string
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
