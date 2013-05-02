/*
Package CloudForest implements ensembles of decision trees for machine learning in pure go (golang).
It includes an implementation of Breiman and Cutler's Random Forest for clasiffication and regression
on heterogenous numerical/catagorical data with missing values.

CloudForest is being developed in the Shumelivich Lab at the Institute for Systems Biology.

Code and Issue tracker can be found at https://github.com/ryanbressler/CloudForest


Speed

When compiled with the default go 1.1 tool chain CloudForest achieves running times similar or
better then implementations in other languages. Using gccgo (4.8.0 at least) results in longer
running times and is not recomended at this time.

CloudForest is especially fast with data that includes lots of binary or low n catagorical data and is
well suited for use on genomic variants.


Goals

CloudForest is intended to provide fast, comprehensible building blocks that can be used
to implement ensembels of decision trees. CloudForest is written in idomatic
go to allow a data scientist to develop and scale new models and analysis quickly
instead of having to modify complex legacy code.

Datastructures and file formats are chosen with use in multi threaded and cluster enviroments
in mind.


Working with Trees

Go's support for first class functions is used to provide a interface to run code as data
is percolated through a tree. This method is flexible enough that it can extend the tree being
analised. Growing a decision tree using Breiman and Cutler's method can be done in an anonymous
function/closure passed to a tree's root node's Recurse method:

	tree.Root.Recurse(func(n *Node, innercases []int) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&canidates, mTry)
			best, impDec := fm.BestSplitter(target, innercases, canidates[:mTry], itter, l, r)
			if best != nil && impDec > minImp {
				//not a leaf node so define the spliter and left and right nodes
				//so recursion will continue
				n.Splitter = best
				n.Pred = ""
				n.Left = new(Node)
				n.Right = new(Node)
				return
			}
		}

		//Leaf node so find the predictive value and set it in n.Pred
		n.Splitter = nil
		n.Pred = target.FindPredicted(innercases)

	}, featurematrix, cases)

This allows a researcher to include whatever additional analaysis they need (importance scores,
proximity etc) in tree growth. The same Recurse method can also be used to analize existing forests
to tabulate scores or extract structure. Utilities like leafcount and errorrate use this
method to tabulate data about the tree in collection objects.


Impurity

Decision tree's are grown with the goal of reducing "Impurity" which is usually defined as Gini
Impurity for catagorical targets or mean squared error for numerical targets. CloudForest grows
trees against the Target interface which allows for alternative definitions of impurity.  L1Target
and RegretTarget impliment L1 norm error regression and a simple cost weighted classification.


Splitting

Repeatedly spliting the data and searching for the best split at each node of a decision tree
are the most computationally intensive parts of decision tree learning and CloudForest includes
optimized (if quirky) code for these areas.

Go's slices are used extensivelly in CloudForest to make it simple to interact with optimized code.
Many previous imlementations of Random Forest have avoided reallocation by reordering data in
place and keeping track of start and end indexes. In go, slices pointing at the same underlying
arrays make this sort of optimization transparent. For example a function like:

	func(s *Splitter) SplitInPlace(fm *FeatureMatrix, cases []int) (l []int, r []int)

can return left and right slices that point to the same underlying array as the origional
slice of cases but these slices should not have their values changed.

Split searching also accepts pointers to slices that will be reset to zero length and reused
without reallocation. These slices won't contain meaningfull data after the search is done but
provide signifigant speed gains. Their use can be seen in the l and r parmaters passed to
BestSplitter in the the tree growing code above and functions that accept them include:

	func (fm *FeatureMatrix) BestSplitter(target Target,
		cases []int,
		canidates []int,
		itter bool,
		l *[]int,
		r *[]int) (s *Splitter, impurityDecrease float64)

	func (f *Feature) BestSplit(target *Feature,
		cases *[]int,
		itter bool,
		l *[]int,
		r *[]int,
		counter *[]int,
		sorter *SortableFeature) (bestNum float64, bestCat int, impurityDecrease float64)

Which accept reusable l, r, counter and sorter objects.

For catagorical predictors, BestSplit will also attempt to inteligently choose between 4
diffrent implementations depending on userinput and the number of catagories.
These include exahustive, random, and iterative searches implemented with bitwise oporations
against int and big.Int dependign on the number of catagories. See BestCatSplit, BestCatSplitIter,
BestCatSplitBig and BestCatSplitIterBig. All numerical predictors are handled by BestNumSplit which
reliest on go's sorting package.


Main Structures

In CloudForest data is stored using the FeatureMatrix struct which contains Features.

The Feature struct  implments storage and methods for both catagorical and numerical data and
calculations of impurity etc and the search for the best split.

The Target interface abstracts the methods of Feature that are needed for a feature to be predictable.
This allows for the implementatiion of alternative types of regression and classification.

Trees are built from Nodes and Splitters and stored within a Forest. Tree has a Grow
implements Brieman and Cutler's method (see extract above) for growing a tree. A GrowForest
method is also provided that implments the rest of the method including sampeling cases
but it may be faster to grow the forest to disk as in the growforest utility.

Prediction and Voteing is done using Tree.Vote and CatBallotBox and NumBallotBox which impliment the
VoteTallyer interface.


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



A stoichastic forest (.sf) file contains a forest of decision trees. The main advantage of this
format as opposed to an established format like json is that an sf file can be written iterativelly
tree by tree and multiple .sf files can be combined with minimal logic required allowing for
massivelly parralel growth of forests with low memory use.

An .sf fileconsists of lines each of which is a comma seperated list of key value pairs. Lines can
designate either a FOREST, TREE, or NODE. Each tree belongs to the preceding forest and each node to
the preciding tree. Nodes must be written in order of increasing depth.

CloudForest generates fiewer fields then rf-ace but requires the following. Other fields will be
ignored

Forest requires forest type (allways RF currently), target and ntrees:

	FOREST=RF|GBT|..,TARGET="$feature_id",NTREES=int

Tree requires only an int and the value is  ignored though the line is needed to designate a new tree:

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

Cloud forest can parse and apply .sf files generated by at least some versions of rf-ace.


Growforest Utility

growforest grows a random forest using the following paramaters which can be listed with -h

	Usage of growforest:
	  -cost="": For catagorical targets, a json string to float map of the cost of falsely identifying each catagory.
	  -cpuprofile="": write cpu profile to file
	  -importance="": File name to output importance.
	  -itterative=true: Use an iterative search for large (n>5) catagorical fearures instead of exahustive/random.
	  -l1=false: Use l1 norm regression (target must be numeric).
	  -leafSize=0: The minimum number of cases on a leaf node. If <=0 will be infered to 1 for clasification 4 for regression.
	  -mTry=0: Number of canidate features for each split. Infered to ceil(swrt(nFeatures)) if <=0.
	  -nSamples=0: The number of cases to sample (with replacment) for each tree grow. If <=0 set to total number of cases
	  -nTrees=100: Number of trees to grow in the predictor.
	  -rfpred="rface.sf": File name to output predictor in rf-aces sf format.
	  -target="": The row header of the target in the feature matrix.
	  -train="featurematrix.afm": AFM formated feature matrix containing training data.


Applyforrest Utility

Not yet implemented.


Errorrate Utility

errorrate calculates the error of a forest vs a testing data set and reports it to standard out

	Usage of errorrate:
	  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
	  -rfpred="rface.sf": A predictor forest as outputed by rf-ace


Leafcount Utility

leafcount outputs counts of case case coocurence on leaf nodes (Brieman's proximity) and counts of the
number of times a feature is used to split a node containing each case (a measure of relative/local
importance).

	Usage of leafcount:
	  -branches="branches.tsv": a case by feature sparse matrix of leaf cooccurance in tsv format
	  -fm="featurematrix.afm": AFM formated feature matrix to use.
	  -leaves="leaves.tsv": a case by case sparse matrix of leaf cooccurance in tsv format
	  -rfpred="rface.sf": A predictor forest as outputed by rf-ace




Refrences

The idea for (and trademakr of the term) Random Forests originated with Leo Brieman and
Adele Cuttler. Their code and paper's can be found at:

http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

All code in CloudForest is origional but some ideas for methods and optimizations were inspired by
rf-ace and R's randomForest package:

https://code.google.com/p/rf-ace/
http://cran.r-project.org/web/packages/randomForest/index.html

Eugene Tuv, Alexander Borisov, George Runger and Kari Torkkola's paper "Feature Selection with
Ensembles, Artificial Variables, and Redundancy Elimination" also deserves mention for its
excellent description and analysis of Random Forests though CloudForest does not (yet?) implement
any of the additional analysis it describes:
http://www.researchgate.net/publication/220320233_Feature_Selection_with_Ensembles_Artificial_Variables_and_Redundancy_Elimination/file/d912f5058a153a8b35.pdf


*/
package CloudForest
