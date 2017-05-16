package CloudForest

import (
	"strconv"
)

const leafFeature = -1

// the evaluator interface implements high performance
// decision tree evaluation strategies.
// The PiecewiseFlatForest and the ContiguousFlatForest
// provide faster analogs of the Predict function
type NumEvaluator interface {
	EvaluateNum(fm *FeatureMatrix) []float64
}

type CatEvaluator interface {
	EvaluateCat(fm *FeatureMatrix) []string
}

type FlatNode struct {
	Feature   int     `json:"feature"`
	Float     float64 `json:"float"`
	Value     string  `json:"value"`
	LeftChild uint32  `json:"leftchild"`
}

type FlatTree struct {
	Nodes  []*FlatNode `json:"nodes"`
	Weight float64
}

func NewFlatTree(t *Tree) *FlatTree {
	f := &FlatTree{
		Nodes:  make([]*FlatNode, 1),
		Weight: adjustWeight(t.Weight),
	}
	f.recurse(t.Root, 0)
	return f
}

func (f *FlatTree) recurse(n *Node, idx uint32) {
	if n.Left == nil && n.Right == nil {
		fl, _ := strconv.ParseFloat(n.Pred, 64)
		f.Nodes[idx] = &FlatNode{
			Feature: leafFeature,
			Float:   fl,
			Value:   n.Pred,
		}
		return
	}
	leftChild := uint32(len(f.Nodes))
	f.Nodes = append(f.Nodes, make([]*FlatNode, 2)...)
	var value string
	var fl float64
	switch x := n.CodedSplit.(type) {
	case float64:
		fl = x
	case int:
		fl = float64(x)
	case string:
		value = x
	}
	f.Nodes[idx] = &FlatNode{
		Feature:   n.Featurei,
		Value:     value,
		Float:     fl,
		LeftChild: leftChild,
	}
	f.recurse(n.Left, leftChild)
	f.recurse(n.Right, leftChild+1)
}

func (f *FlatTree) EvaluateNum(fm *FeatureMatrix) []float64 {
	sz := fm.Data[0].Length()
	preds := make([]float64, sz)
	for i := 0; i < sz; i++ {
		current := uint32(0)
		for {
			n := f.Nodes[current]
			// leaf node
			if n.Feature == leafFeature {
				preds[i] = n.Float * f.Weight
				break
			}
			switch f := fm.Data[n.Feature].(type) {
			case *DenseNumFeature:
				val := f.NumData[i]
				splitValue := n.Float
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

func (f *FlatTree) EvaluateCat(fm *FeatureMatrix) []string {
	sz := fm.Data[0].Length()
	preds := make([]string, sz)
	for i := 0; i < sz; i++ {
		current := uint32(0)
		for {
			n := f.Nodes[current]
			// leaf node
			if n.Feature == leafFeature {
				preds[i] = n.Value
				break
			}
			switch f := fm.Data[n.Feature].(type) {
			case *DenseNumFeature:
				val := f.NumData[i]
				splitValue := n.Float
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
	Trees     []*FlatTree `json:"trees"`
	Intercept float64
}

func NewPiecewiseFlatForest(forest *Forest) *PiecewiseFlatForest {
	p := &PiecewiseFlatForest{
		Trees:     make([]*FlatTree, len(forest.Trees)),
		Intercept: forest.Intercept,
	}
	for i, n := range forest.Trees {
		p.Trees[i] = NewFlatTree(n)
	}
	return p
}

func (p *PiecewiseFlatForest) EvaluateNum(fm *FeatureMatrix) []float64 {
	sz := fm.Data[0].Length()
	n := float64(len(p.Trees))
	preds := make([]float64, sz)
	for i := range preds {
		preds[i] = p.Intercept
	}

	for _, tree := range p.Trees {
		for i, pred := range tree.EvaluateNum(fm) {
			preds[i] += pred / n
		}
	}
	return preds
}

func (p *PiecewiseFlatForest) EvaluateCat(fm *FeatureMatrix) []string {
	sz := fm.Data[0].Length()
	bb := NewCatBallotBox(sz)
	for _, tree := range p.Trees {
		for i, pred := range tree.EvaluateCat(fm) {
			bb.Vote(i, pred, 1.0)
		}
	}

	preds := make([]string, sz)
	for i := 0; i < sz; i++ {
		preds[i] = bb.Tally(i)
	}
	return preds
}

type ContiguousFlatForest struct {
	Roots     []uint32
	Nodes     []*FlatNode
	Weights   []float64
	Intercept float64
}

func NewContiguousFlatForest(forest *Forest) *ContiguousFlatForest {
	var roots []uint32
	var nodes []*FlatNode
	var weights []float64
	for _, tree := range forest.Trees {
		idx := uint32(len(nodes))
		roots = append(roots, idx)
		weights = append(weights, adjustWeight(tree.Weight))
		for _, node := range NewFlatTree(tree).Nodes {
			node.LeftChild += idx
			nodes = append(nodes, node)
		}
	}
	return &ContiguousFlatForest{
		Roots:     roots,
		Nodes:     nodes,
		Weights:   weights,
		Intercept: forest.Intercept,
	}
}

func (c *ContiguousFlatForest) EvaluateNum(fm *FeatureMatrix) []float64 {
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
					result += n.Float
					break
				}
				switch f := fm.Data[n.Feature].(type) {
				case *DenseNumFeature:
					val := f.NumData[i]
					splitValue := n.Float
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
		preds[i] = (result / float64(len(c.Roots))) * c.Weights[i]
		preds[i] += c.Intercept
	}
	return preds
}

func (c *ContiguousFlatForest) EvaluateCat(fm *FeatureMatrix) []string {
	sz := fm.Data[0].Length()
	bb := NewCatBallotBox(sz)

	for i := 0; i < sz; i++ {
		for _, root := range c.Roots {
			current := root
			for {
				n := c.Nodes[current]
				if n.Feature == leafFeature {
					// im a leaf
					bb.Vote(i, n.Value, 1.0)
					break
				}
				switch f := fm.Data[n.Feature].(type) {
				case *DenseNumFeature:
					val := f.NumData[0]
					splitValue := n.Float
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
	}

	preds := make([]string, sz)
	for i := 0; i < sz; i++ {
		preds[i] = bb.Tally(i)
	}
	return preds
}

func adjustWeight(x float64) float64 {
	if x <= 0.0 {
		return 1.0
	}
	return x
}
