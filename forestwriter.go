package CloudForest

import (
	"fmt"
	"io"
	"strings"
)

/*
ForestWriter wraps an io writer with functionality to write forests either with one
call to WriteForest or incrementally using WriteForestHeader and WriteTree.
ForestWriter saves a forest in .sf format; see the package doc's in doc.go for
full format details.
It won't include fields that are not use by CloudForest.
*/
type ForestWriter struct {
	w io.Writer
}

/*NewForestWriter returns a pointer to a new ForestWriter. */
func NewForestWriter(w io.Writer) *ForestWriter {
	return &ForestWriter{w}
}

//WriteForest writes an entire forest including all headers.
func (fw *ForestWriter) WriteForest(forest *Forest) {
	fw.WriteForestHeader(forest.Target, len(forest.Trees))
	for i, tree := range forest.Trees {
		fw.WriteTree(tree, i)
	}
}

//WriteTree writes an entire Tree including the header.
func (fw *ForestWriter) WriteTree(tree *Tree, ntree int) {
	fw.WriteTreeHeader(ntree)
	fw.WriteNodeAndChildren(tree.Root, "*")
}

//WriteForestHeader writes only the header of a forest.
func (fw *ForestWriter) WriteForestHeader(target string, ntrees int) {
	fmt.Fprintf(fw.w, "FOREST=RF,TARGET=%v,NTREES=%v\n", target, ntrees)
}

//WrieTreeHeader writes only the header line for a tree.
func (fw *ForestWriter) WriteTreeHeader(ntree int) {
	fmt.Fprintf(fw.w, "TREE=%v\n", ntree)
}

//WriteNodeAndChildren recursively writes out the target node and all of its children.
//WriteTree is preferred for most use cases.
func (fw *ForestWriter) WriteNodeAndChildren(n *Node, path string) {

	fw.WriteNode(n, path)
	if n.Splitter != nil && n.Left != nil {
		fw.WriteNodeAndChildren(n.Left, path+"L")
	}
	if n.Splitter != nil && n.Right != nil {
		fw.WriteNodeAndChildren(n.Right, path+"R")
	}
	if n.Splitter != nil && n.Missing != nil {
		fw.WriteNodeAndChildren(n.Right, path+"M")
	}

}

//WriteNode writes a single node but not it's children. WriteTree will be used more
//often but WriteNode can be used to grow a large tree directly to disk without
//storing it in memory.
func (fw *ForestWriter) WriteNode(n *Node, path string) {
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
			left := fw.DescribeMap(n.Splitter.Left)
			node += fmt.Sprintf(",SPLITTERTYPE=CATEGORICAL,LVALUES=%v", left)
		}
	}
	fmt.Fprintln(fw.w, node)
}

//DescribeMap serializes the "left" map of a categorical splitter.
func (fw *ForestWriter) DescribeMap(input map[string]bool) string {
	keys := make([]string, 0)
	for k := range input {
		keys = append(keys, k)
	}
	return "\"" + strings.Join(keys, ":") + "\""
}
