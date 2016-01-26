package CloudForest

// ScikitNode
// cdef struct Node:
//     # Base storage structure for the nodes in a Tree object

//     SIZE_t left_child                    # id of the left child of the node
//     SIZE_t right_child                   # id of the right child of the node
//     SIZE_t feature                       # Feature used for splitting the node
//     DOUBLE_t threshold                   # Threshold value at the node
//     DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
//     SIZE_t n_node_samples                # Number of samples at the node
//     DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

type ScikitNode struct {
	LeftChild            int     `json:"left_child"`
	RightChild           int     `json:"right_child"`
	Feature              int     `json:"feature"`
	Threshold            float64 `json:"threshold"`
	Impurity             float64 `json:"impurity"`                //TODO(ryan): support this?
	NNodeSamples         int     `json:"n_node_samples"`          //TODO(ryan): support this?
	WeightedNNodeSamples float64 `json:"weighted_n_node_samples"` //TODO(ryan): support this?
}

// AnnotatedTree represents a decision tree in the memory format used by scikit learn.
// cdef class Tree:
//     # The Tree object is a binary tree structure constructed by the
//     # TreeBuilder. The tree structure is used for predictions and
//     # feature importances.

//     # Input/Output layout
//     cdef public SIZE_t n_features        # Number of features in X
//     cdef SIZE_t* n_classes               # Number of classes in y[:, k]
//     cdef public SIZE_t n_outputs         # Number of outputs in y
//     cdef public SIZE_t max_n_classes     # max(n_classes)

//     # Inner structures: values are stored separately from node structure,
//     # since size is determined at runtime.
//     cdef public SIZE_t max_depth         # Max depth of the tree
//     cdef public SIZE_t node_count        # Counter for node IDs
//     cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
//     cdef Node* nodes                     # Array of nodes
//     cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
//     cdef SIZE_t value_stride             # = n_outputs * max_n_classes
type ScikitTree struct {
	NFeatures   int           `json:"n_features"`
	NClasses    []int         `json:"n_classes"`
	NOutputs    int           `json:"n_outputs"`     //TODO(ryan): support other values
	MaxNClasses int           `json:"max_n_classes"` //TODO(ryan): support other values
	MaxDepth    int           `json:"max_depth"`
	NodeCount   int           `json:"node_count"`
	Capacity    int           `json:"capacity"`
	Nodes       []ScikitNode  `json:"nodes"`
	Value       [][][]float64 `json:"value"` //TODO(ryan): support actual values
	ValueStride int           `json:"value_stride"`
}

func NewScikitTree(nFeatures int) *ScikitTree {
	tree := &ScikitTree{
		NFeatures:   nFeatures,
		NClasses:    []int{2},
		NOutputs:    1,
		MaxNClasses: 2,
		MaxDepth:    0,
		NodeCount:   0,
		Capacity:    0,
		Nodes:       make([]ScikitNode, 0),
		Value:       make([][][]float64, 0),
		ValueStride: 0}

	return tree
}

// BuildScikkitTree currentelly only builds the split threshold and node structure of a sickit tree from a
// Cloudforest tree specified by root node
func BuildScikitTree(depth int, n *Node, sktree *ScikitTree) {
	if depth > sktree.MaxDepth {
		sktree.MaxDepth = depth
	}
	depth++
	sktree.NodeCount++
	sktree.Capacity++
	skn := ScikitNode{}
	pos := len(sktree.Nodes)
	// We can't use a pointer here because the array will move and we're building this as an array
	// of structs for sklearn memory compatibility later so we use a pos.
	sktree.Nodes = append(sktree.Nodes, skn)
	if n.Splitter != nil {
		sktree.Nodes[pos].Feature = n.Featurei
		sktree.Nodes[pos].Threshold = n.Splitter.Value
		sktree.Nodes[pos].LeftChild = sktree.NodeCount
		BuildScikitTree(depth, n.Left, sktree)
		sktree.Nodes[pos].RightChild = sktree.NodeCount
		BuildScikitTree(depth, n.Right, sktree)

	} else {
		// Leaf node
		sktree.Nodes[pos].LeftChild = -1
		sktree.Nodes[pos].RightChild = -1
	}
}
