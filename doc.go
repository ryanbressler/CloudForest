/*
Package CloudForest implements decision trees for machine learning and includes a pure go 
implementation of Breiman and Cutler's Random Forest for clasiffication and regression on 
heterogenous numerical/catagorical data with missing values.

Easy hackability is a main design goal. CloudForest provides basic building blocks that
allow researchers quickly implement new and existing decision tree based methods. For example 
growing a decision tree using Brieman's method is as easy as:

	func (t *Tree) Grow(fm *FeatureMatrix, target *Feature, cases []int, canidates []int, mTry int, leafSize int) {
		t.Root.Recurse(func(n *Node, innercases []int) {

			if leafSize < len(innercases) {
				SampleFirstN(&canidates, mTry)
				best, impDec := target.BestSplitter(fm, innercases, canidates[:mTry])
				if best != nil && impDec > 0.0000001 {
					//not a leaf node so define the spliter and left and right nodes
					//so recursion will continue
					n.Splitter = best
					n.Left = &Node{nil, nil, "", nil}
					n.Right = &Node{nil, nil, "", nil}
					return
				}
			}

			//Leaf node so find the predictive value and set it in n.Pred
			n.Splitter = nil
			n.Pred = target.FindPredicted(innercases)

		}, fm, cases)
	}

This node.Recurse function (which might be better named percolate or climb) can also be used 
to apply an existing forest to new data or analize a forests interaction with data to calcualte 
imporatnce/proximity etc. 

In CloudForest data is stored using the FeatureMatrix struct which contains Features. 

The Featurestruct currently implments both catagorical and numerical but may be abstracted to an
Interface allowing seperate implementations more feature types in the future. It is responsible for
feature type specific code including calculations of impurity etc and searching for best splits.

Trees are built from Nodes and Splitters and stored within a Forest.

Voteing is done using CatBallotBox and NumBallotBox which impliment the Voter interface.

Utilities

growforest grows a random forest using the following paramaters

	  -cpuprofile="": write cpu profile to file
	  -leafSize=0: The minimum number of cases on a leaf node. If <=0 will be infered to 1 for clasification 4 for regression.
	  -mTry=0: Number of canidate features for each split. Infered to ceil(swrt(nFeatures)) if <=0.
	  -nSamples=0: The number of cases to sample (with replacment) for each tree grow. If <=0 set to total number of cases
	  -nTrees=100: Number of trees to grow in the predictor.
	  -rfpred="rface.sf": File name to output predictor in rf-aces sf format.
	  -target="": The row header of the target in the feature matrix.
	  -train="featurematrix.afm": AFM formated feature matrix containing training data.




File Formats

CloudForest borrows the anotated feature matrix (.afm) and stoicastic forest (.sf) file formats and 
from rf-ace which can be found at https://code.google.com/p/rf-ace/


An .afm file is a tab deliminated file with column and row headers. Columns represent cases and rows
represent features. A row header/feature id includes a prefix to specify the feature type

	"N:" Prefix for numerical feature id.
	"C:" Prefix for catagorical feature id.
	"B:" Prefix for boolean feature id.

Catagorical and boolean features use strings for their catagorie labels. Missing values are represented 
by "?","nan","na", or "null" (case insensative). A short example:

	featureid	case1	case2	case3
	N:NumF1	0.0	.1	na
	C:CatF2 red	red	green

An .sf file:

	FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800,CATEGORIES="",SHRINKAGE=0
	TREE=0
	NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false",RVALUES="true"
	NODE=*L,PRED=3.75
	NODE=*R,PRED=1


CloudForest was developed by Ryan Bressler at the Institute for Systems Biology and is Released 
under a modified bsd style license. Code can be found at https://github.com/ryanbressler/CloudForest

CloudForest borrows file formats and ideas from rf-ace which can be found at https://code.google.com/p/rf-ace/

*/
package CloudForest
