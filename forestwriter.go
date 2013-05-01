package CloudForest

import (
	"fmt"
	"io"
)

//Not fully implemented yet.
type ForestWriter struct {
	w io.Writer
}

/*NewForestWriter save's a forest in rf-ace's "stoicastic forest" sf format
It won't include fields that are not use by cloud forest.
Start of an example file:

	FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800,CATEGORIES="",SHRINKAGE=0
	TREE=0
	NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false",RVALUES="true"
	NODE=*L,PRED=3.75
	NODE=*R,PRED=1

Node should be a path the form *LRL where * indicates the root L and R indicate Left and Right.*/
func NewForestWriter(w io.Writer) *ForestWriter {
	return &ForestWriter{w}
}

func (fw *ForestWriter) WriteForest(forest *Forest) {
	fw.WriteForestHeader(forest.Target, len(forest.Trees))
	for i, tree := range forest.Trees {
		fw.WriteTree(tree, i)
	}
}

func (fw *ForestWriter) WriteTree(tree *Tree, ntree int) {
	fw.WriteTreeHeader(ntree)
	fw.WriteNodeAndChildren(tree.Root, "*")
}

func (fw *ForestWriter) WriteForestHeader(target string, ntrees int) {
	fmt.Fprintf(fw.w, "FOREST=RF,TARGET=%v,NTREES=%v\n", target, ntrees)
}

func (fw *ForestWriter) WriteTreeHeader(ntree int) {
	fmt.Fprintf(fw.w, "TREE=%v\n", ntree)
}

func (fw *ForestWriter) WriteNodeAndChildren(n *Node, path string) {
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
	fmt.Fprintln(fw.w, node)

	if n.Splitter != nil && n.Left != nil {
		fw.WriteNodeAndChildren(n.Left, path+"L")
	}
	if n.Splitter != nil && n.Right != nil {
		fw.WriteNodeAndChildren(n.Right, path+"R")
	}

}
