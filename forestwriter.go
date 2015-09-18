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
	if forest.Intercept != 0.0 {
		fw.WriteForestHeader(0, forest.Target, forest.Intercept)
	}
	for i, tree := range forest.Trees {
		fw.WriteTree(tree, i)
	}
}

//WriteTree writes an entire Tree including the header.
func (fw *ForestWriter) WriteTree(tree *Tree, ntree int) {
	fw.WriteTreeHeader(ntree, tree.Target, tree.Weight)
	fw.WriteNodeAndChildren(tree.Root, "*")
}

//WrieTreeHeader writes only the header line for a tree.
func (fw *ForestWriter) WriteTreeHeader(ntree int, target string, weight float64) {
	weightterm := ""
	if weight >= 0.0 {
		weightterm = fmt.Sprintf(",WEIGHT=%v", weight)
	}
	fmt.Fprintf(fw.w, "TREE=%v,TARGET=\"%v\"%v\n", ntree, target, weightterm)
}

//WrieTreeHeader writes only the header line for a tree.
func (fw *ForestWriter) WriteForestHeader(nforest int, target string, intercept float64) {
	interceptterm := ""
	if intercept != 0.0 {
		interceptterm = fmt.Sprintf(",INTERCEPT=%v", intercept)
	}
	fmt.Fprintf(fw.w, "FOREST=%v,TARGET=\"%v\"%v\n", nforest, target, interceptterm)
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
