package CloudForest

import (
	"fmt"
	"strconv"
)

const leafFeature = -1

// the evaluator interface implements high performance
// decision tree evaluation strategies.
// The PiecewiseFlatForest and the ContiguousFlatForest
// provide faster analogs of the Predict function
type Evaluator interface {
	Evaluate(fm *FeatureMatrix) []float64
}

type FlatNode struct {
	Feature   int    `json:"feature"`
	Value     string `json:"value"`
	LeftChild uint32 `json:"leftchild"`
}

type FlatTree struct {
	Nodes []*FlatNode `json:"nodes"`
}

func NewFlatTree(root *Node) *FlatTree {
	f := &FlatTree{make([]*FlatNode, 1)}
	f.recurse(root, 0)
	return f
}

func (f *FlatTree) recurse(n *Node, idx uint32) {
	if n.Left == nil && n.Right == nil {
		f.Nodes[idx] = &FlatNode{
			Feature: leafFeature,
			Value:   n.Pred,
		}
		return
	}
	leftChild := uint32(len(f.Nodes))
	f.Nodes = append(f.Nodes, make([]*FlatNode, 2)...)
	var value string
	switch x := n.CodedSplit.(type) {
	case float64, int:
		value = fmt.Sprintf("%v", x)
	case string:
		value = x
	}
	f.Nodes[idx] = &FlatNode{
		Feature:   n.Featurei,
		Value:     value,
		LeftChild: leftChild,
	}
	f.recurse(n.Left, leftChild)
	f.recurse(n.Right, leftChild+1)
}

func (f *FlatTree) Evaluate(fm *FeatureMatrix) []float64 {
	sz := fm.Data[0].Length()
	preds := make([]float64, sz)
	for i := 0; i < sz; i++ {
		current := uint32(0)
		for {
			n := f.Nodes[current]
			// leaf node
			if n.Feature == leafFeature {
				val, _ := strconv.ParseFloat(n.Value, 64)
				preds[i] = val
				break
			}
			switch f := fm.Data[n.Feature].(type) {
			case *DenseNumFeature:
				val := f.NumData[i]
				splitValue, _ := strconv.ParseFloat(n.Value, 64)
				if val < splitValue {
					current = n.LeftChild
				} else {
					current = n.LeftChild + 1
				}
			case *DenseCatFeature:
				val := f.GetStr(i)
				splitValue := n.Value
				if val == splitValue {
					current = n.LeftChild
				} else {
					current = n.LeftChild + 1
				}
			}
		}
	}
	return preds
}

type PiecewiseFlatForest struct {
	Trees []*FlatTree `json:"trees"`
}

func NewPiecewiseFlatForest(forest *Forest) *PiecewiseFlatForest {
	p := &PiecewiseFlatForest{make([]*FlatTree, len(forest.Trees))}
	for i, n := range forest.Trees {
		p.Trees[i] = NewFlatTree(n.Root)
	}
	return p
}

func (p *PiecewiseFlatForest) Evaluate(fm *FeatureMatrix) []float64 {
	sz := fm.Data[0].Length()
	n := float64(len(p.Trees))
	preds := make([]float64, sz)
	for _, tree := range p.Trees {
		for i, pred := range tree.Evaluate(fm) {
			preds[i] += pred / n
		}
	}
	return preds
}

type ContiguousFlatForest struct {
	Roots []uint32
	Nodes []*FlatNode
}

func NewContiguousFlatForest(forest *Forest) *ContiguousFlatForest {
	var roots []uint32
	var nodes []*FlatNode
	for _, tree := range forest.Trees {
		idx := uint32(len(nodes))
		roots = append(roots, idx)
		for _, node := range NewFlatTree(tree.Root).Nodes {
			node.LeftChild += idx
			nodes = append(nodes, node)
		}
	}
	return &ContiguousFlatForest{
		Roots: roots,
		Nodes: nodes,
	}
}

func (c *ContiguousFlatForest) Evaluate(fm *FeatureMatrix) []float64 {
	sz := fm.Data[0].Length()
	preds := make([]float64, sz)
	for i := 0; i < sz; i++ {
		result := 0.0
		for _, root := range c.Roots {
			current := root
			for {
				n := c.Nodes[current]
				if n.Feature == leafFeature {
					// im a leaf
					val, _ := strconv.ParseFloat(n.Value, 64)
					result += val
					break
				}
				switch f := fm.Data[n.Feature].(type) {
				case *DenseNumFeature:
					val := f.NumData[0]
					splitValue, _ := strconv.ParseFloat(n.Value, 64)
					if val < splitValue {
						current = n.LeftChild
					} else {
						current = n.LeftChild + 1
					}
				case *DenseCatFeature:
					val := f.GetStr(0)
					splitValue := n.Value
					if val == splitValue {
						current = n.LeftChild
					} else {
						current = n.LeftChild + 1
					}
				}
			}
		}
		preds[i] = result / float64(len(c.Roots))
	}
	return preds
}
