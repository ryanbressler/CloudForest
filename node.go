package CloudForest

import (
	"fmt"
	"io"
)

//Recurssable defines a function signature for functions that can be called at every
//down stream node of a tree as Node.Recurse recurses up the tree. The function should
//have two paramaters, the current node and an array of ints specifying the cases that
//have not been split away.
type Recursable func(*Node, []int)

//A node of a decision tree.
//Pred is a string containg either the catagory or a representation of a float
//(less then ideal)
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
		ls, rs := n.Splitter.SplitInPlace(fm, cases)
		n.Left.Recurse(r, fm, ls)
		n.Right.Recurse(r, fm, rs)
	}
}

func (n *Node) Write(w io.Writer, path string) {
	node := fmt.Sprintf("NODE=%v", path)
	if n.Pred != "" {
		node += fmt.Sprintf(",PRED=%v", n.Pred)
	}

	if n.Splitter != nil {
		node += fmt.Sprintf(",SPLITTER=%v", n.Splitter.Feature)
		switch n.Splitter.Numerical {
		case true:
			node += fmt.Sprintf(",SPLITTERTYPE=NUMERICAL,LVALUES=%v,RVALUES=%v", n.Splitter.Value, n.Splitter.Value)
		case false:
			left := n.Splitter.DescribeMap(n.Splitter.Left)
			node += fmt.Sprintf(",SPLITTERTYPE=CATEGORICAL,LVALUES=%v", left)
		}
	}
	fmt.Fprintln(w, node)
	if n.Left != nil {
		n.Left.Write(w, path+"L")
	}
	if n.Right != nil {
		n.Right.Write(w, path+"R")
	}

}
