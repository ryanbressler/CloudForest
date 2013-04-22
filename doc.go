/*CloudForest implements hackable data structures and analysis for working with ensembels
of decision trees including random forests.

Currently cloud forest supports parsing and analizing forests grown with rf-ace. In the
near future it will support growing forests in pure go.

"Hackability" is a main design goal in that CloudForest provides basic building blocks that
allow researchers quickly implement novel decision tree based methods. For example growing
a decision tree with a simple set of rules will be as easy as:

	func (t *Tree) Grow(fm *FeatureMatrix, target *Feature, cases []int, mTry int, leafSize int) {
		t.Root.Recurse(func(n *Node, cases []int) {
			if leafSize < len(cases) {
				best := target.BestSplitter(fm, cases, mTry)
				//not a leaf node so define the spliter and left and right nodes
				//so recursion will continue up the tree
				n.Splitter = best
				n.Left = new(Node)
				n.Right = new(Node)
				return
			}

			//Leaf node so find the predictive value and set it in n.Pred
			n.Pred = target.FindPredicted(cases)

		}, fm, cases)
	}


*/
package CloudForest
