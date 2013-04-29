/*
Package CloudForest implements decision trees for machine learning and includes a pure go
implementation of Breiman and Cutler's Random Forest for clasiffication and regression on
heterogenous numerical/catagorical data with missing values.

When compiled with the default go 1.1 tool chain CloudForest achieves running times similar or
better then previous implementations. Using gccgo 4.8.0 results in longer runnign times and
is not recomended.

CloudForest is being developed by Ryan Bressler in the Shumelivich Lab at the Institute
for Systems Biology and is released under a modified BSD style license.

Code and the Bug tracker can be found at https://github.com/ryanbressler/CloudForest


Philosophy and Goals

CloudForest is intended to provide fast, comprehensible building blocks that can be used
to implement ensembels of decision trees. CloudForest is written in (somewhat) idomatic
go to allow a programer to implement and scale new models and analysis quickly instead
of having to modify complex code.

Go's support for first class functions is key to CloudForests flexiblity. For example
growing a decision tree using Breiman and Cutler's method is as easy as:

	tree.Root.Recurse(func(n *Node, innercases []int) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&canidates, mTry)
			best, impDec := target.BestSplitter(fm, innercases, canidates[:mTry], itter, l, r)
			if best != nil && impDec > minImp {
				//not a leaf node so define the spliter and left and right nodes
				//so recursion will continue
				n.Splitter = best
				n.Pred = ""
				n.Left = &Node{nil, nil, "", nil}
				n.Right = &Node{nil, nil, "", nil}
				return
			}
		}

		//Leaf node so find the predictive value and set it in n.Pred
		n.Splitter = nil
		n.Pred = target.FindPredicted(innercases)

	}, featurematrix, cases)

This node.Recurse function (which might be better named percolate or climb) can also be used
to apply an existing forest to new data or analize a forests interaction with data to calcualte
imporatnce/proximity etc.

Go's slices are used extensivelly in CloudForest to make it simple to interact with optimized code.
Many previous imlementations of Random Forest have avoided reallocation by reordering data in
place and keeping track of start and end indexes. In go, slices pointing at the same underlying
arrays make this sort of optimization transparent. For example Splitter.SplitInPlace returns
left and right slices that point to the same underlying array as the origional slice of cases.

Code that is called repeatedly during training/tree growth also accepts pointers to slices that
will be reset to zero length and reused without reallocation proving signifgant speed gains.

Future goals include a full set of comand line utilites and the implementation of methods like
extra random trees and gradiant boosting trees. Internally we will move towards the abstraction
of a data feature to an interface to allow extension to non catacagroical and numeric features
and the further abstraction of collection objects to support paralelization across many machines.


Overview

In CloudForest data is stored using the FeatureMatrix struct which contains Features.

The Feature struct  implments both catagorical and numerical data and is responsible for
feature type specific code including calculations of impurity etc and searching for best splits.
It may be abstracted to an interface in the future.

Trees are built from Nodes and Splitters and stored within a Forest. Tree has a Grow method that
implements Brieman and Cutler's method (see extract above). A GrowForest method is also provided
but it may be faster to grow the forest to disk as in the growforest utility.

Prediction/Voteing is done using CatBallotBox and NumBallotBox which impliment the VoteTallyer
interface.


Utilities

All utilities can be ran with -h to report ussage.

growforest grows a random forest using the following paramaters

	Usage of growforest:
	  -cpuprofile="": write cpu profile to file
	  -leafSize=0: The minimum number of cases on a leaf node. If <=0 will be infered to 1 for clasification 4 for regression.
	  -mTry=0: Number of canidate features for each split. Infered to ceil(swrt(nFeatures)) if <=0.
	  -nSamples=0: The number of cases to sample (with replacment) for each tree grow. If <=0 set to total number of cases
	  -nTrees=100: Number of trees to grow in the predictor.
	  -rfpred="rface.sf": File name to output predictor in rf-aces sf format.
	  -target="": The row header of the target in the feature matrix.
	  -train="featurematrix.afm": AFM formated feature matrix containing training data.


errorrate calculates the error of a forest vs a testing data set and reports it to standard out

	Usage of errorrate:
	  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
	  -rfpred="rface.sf": A predictor forest as outputed by rf-ace



File Formats

CloudForest borrows the anotated feature matrix (.afm) and stoicastic forest (.sf) file formats
from Timo Erkkila's rf-ace which can be found at https://code.google.com/p/rf-ace/


An anotated feature matrix (.afm) file is a tab deliminated file with column and row headers. Columns represent cases and rows
represent features. A row header/feature id includes a prefix to specify the feature type

	"N:" Prefix for numerical feature id.
	"C:" Prefix for catagorical feature id.
	"B:" Prefix for boolean feature id.

Catagorical and boolean features use strings for their catagorie labels. Missing values are represented
by "?","nan","na", or "null" (case insensative). A short example:

	featureid	case1	case2	case3
	N:NumF1	0.0	.1	na
	C:CatF2 red	red	green


A stoichastic forest (.sf) file consists of a lines each of which is a comma seperated list of
key value pairs. Lines can designate either a FOREST, TREE, or NODE. CloudForest generates fiewer fields then
rf-ace but requires the following. Other fields will be ignored

Forest require forest type (allways RF currently), target and ntrees:

	FOREST=[RF|GBT|..],TARGET="$feature_id",NTREES=int

Tree requires only an int and the value is actually ignored though the line is needed to designate a new tree:

	TREE=int

Node requires a path encoded so that the root node is specified by "*" and each split left or right as "L" or "R".
Leaf nodes should also define PRED such as "PRED=1.5" or "PRED=red". Splitter nodes should define SPLITTER with
a feature id inside of double quotes, SPLITTERTYPE=[CATEGORICAL|NUMERICAL] and a LVALUE term which can be either
a float inside of double quotes representing the highest value sent left or a ":" seperated list of catagorical
values sent left.

	NODE=$path,PRED=[float|string],SPLITTER="$feature_id",SPLITTERTYPE=[CATEGORICAL|NUMERICAL] LVALUES="[float|: seperated list"


An example .sf file:

	FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800
	TREE=0
	NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false"
	NODE=*L,PRED=3.75
	NODE=*R,PRED=1

Cloud forest can parse and apply .sf files generated by at least some versions of rf-ace



*/
package CloudForest
