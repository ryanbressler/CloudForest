/*
Package CloudForest implements ensembles of decision trees for machine
learning in pure Go (golang.org). It includes implementations of Breiman
and Cutler's Random Forest for classification and regression on heterogeneous
numerical/categorical data with missing values and several related algorithms
including entropy and cost driven classification, L1 regression and feature
selection with artificial contrasts and hooks for modifying the algorithms
for your needs.

Command line utilities to grow, apply and canalize forests are provided in sub
directories.

CloudForest is being developed in the Shumelivich Lab at the Institute for Systems
Biology.


Documentation has been generated with godoc and can be viewed live at:
http://godoc.org/github.com/ryanbressler/CloudForest

Pull requests and bug reports are welcome; Code Repo and Issue tracker can be found at:
https://github.com/ryanbressler/CloudForest


Goals

CloudForest is intended to provide fast, comprehensible building blocks that can
be used to implement ensembles of decision trees. CloudForest is written in Go to
allow a data scientist to develop and scale new models and analysis quickly instead
of having to modify complex legacy code.

Data structures and file formats are chosen with use in multi threaded and cluster
environments in mind.


Working with Trees

Go's support for function types is used to provide a interface to run code as data
is percolated through a tree. This method is flexible enough that it can extend the tree being
analyzed. Growing a decision tree using Breiman and Cutler's method can be done in an anonymous
function/closure passed to a tree's root node's Recurse method:

	t.Root.Recurse(func(n *Node, innercases []int) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&candidates, mTry)
			best, impDec := fm.BestSplitter(target, innercases, candidates[:mTry], allocs)
			if best != nil && impDec > minImp {
				//not a leaf node so define the splitter and left and right nodes
				//so recursion will continue
				n.Splitter = best
				n.Pred = ""
				n.Left = new(Node)
				n.Right = new(Node)

				return
			}
		}

This allows a researcher to include whatever additional analysis they need (importance scores,
proximity etc) in tree growth. The same Recurse method can also be used to analyze existing forests
to tabulate scores or extract structure. Utilities like leafcount and errorrate use this
method to tabulate data about the tree in collection objects.


Alternative Impurities

Decision tree's are grown with the goal of reducing "Impurity" which is usually defined as Gini
Impurity for categorical targets or mean squared error for numerical targets. CloudForest grows
trees against the Target interface which allows for alternative definitions of impurity. CloudForest
includes several alternative targets:

 EntropyTarget : For use in entropy minimizing classification
 RegretTarget  : For use in classification driven by differing costs in miscategorization.
 L1Target      : For use in L1 norm error regression (which may be less sensitive to outliers).


Efficient Splitting

Repeatedly splitting the data and searching for the best split at each node of a decision tree
are the most computationally intensive parts of decision tree learning and CloudForest includes
optimized code to perform these tasks.

Go's slices are used extensively in CloudForest to make it simple to interact with optimized code.
Many previous implementations of Random Forest have avoided reallocation by reordering data in
place and keeping track of start and end indexes. In go, slices pointing at the same underlying
arrays make this sort of optimization transparent. For example a function like:

	func(s *Splitter) SplitInPlace(fm *FeatureMatrix, cases []int) (l []int, r []int)

can return left and right slices that point to the same underlying array as the origional
slice of cases but these slices should not have their values changed.

Functions used while searching for the best split also accepts pointers to reusable slices and
structs to maximize speed by keeping memory allocations to a minimum. BestSplitAllocs contains
pointers to these items and its use can be seen in functions like:

	func (fm *FeatureMatrix) BestSplitter(target Target,
		cases []int,
		candidates []int,
		allocs *BestSplitAllocs) (s *Splitter, impurityDecrease float64)

	func (f *Feature) BestSplit(target Target,
		cases *[]int,
		parentImp float64,
		allocs *BestSplitAllocs) (bestNum float64, bestCat int, bestBigCat *big.Int, impurityDecrease float64)


For categorical predictors, BestSplit will also attempt to intelligently choose between 4
different implementations depending on user input and the number of categories.
These include exhaustive, random, and iterative searches for the best combination of categories
implemented with bitwise operations against int and big.Int. See BestCatSplit, BestCatSplitIter,
BestCatSplitBig and BestCatSplitIterBig.

All numerical predictors are handled by BestNumSplit which
relies on go's sorting package.


Missing Values

By default cloud forest uses a fast heuristic for missing values. When proposing a split on a feature
with missing data the missing cases are removed and the impurity value is corrected to use three way impurity
which reduces the bias towards features with lots of missing data:

								I(split) = p(l)I(l)+p(r)I(r)+p(m)I(m)

Missing values in the target variable are left out of impurity calculations.

This provided generally good results at a fraction of the computational costs of imputing data.

Optionally, feature.ImputeMissing or featurematrixImputeMissing can be called before forest growth
to impute missing values to the feature mean/mode which Brieman [2] suggests as a fast method for
imputing values.

This forest could also be analyzed for proximity (using leafcount or tree.GetLeaves) to do the
more accurate proximity weighted imputation Brieman describes.

Experimental support is provided for 3 way splitting which splits missing cases onto a third branch.
[2] This has so far yielded mixed results in testing.

At some point in the future support may be added for local imputing of missing values during tree growth
as described in [3]

[1] http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1

[2] https://code.google.com/p/rf-ace/

[3] http://projecteuclid.org/DPubS?verb=Display&version=1.0&service=UI&handle=euclid.aoas/1223908043&page=record

Importance and Contrasts

Variable Importance in CloudForest is calculated as the mean decrease in impurity over all of
the splits made using a feature.

To provide a baseline for evaluating importance, artificial contrast features can be used by
including shuffled copies of existing features.


Main Structures

In CloudForest data is stored using the FeatureMatrix struct which contains Features.

The Feature struct  implements storage and methods for both categorical and numerical data and
calculations of impurity etc and the search for the best split.

The Target interface abstracts the methods of Feature that are needed for a feature to be predictable.
This allows for the implementation of alternative types of regression and classification.

Trees are built from Nodes and Splitters and stored within a Forest. Tree has a Grow
implements Brieman and Cutler's method (see extract above) for growing a tree. A GrowForest
method is also provided that implements the rest of the method including sampling cases
but it may be faster to grow the forest to disk as in the growforest utility.

Prediction and Voting is done using Tree.Vote and CatBallotBox and NumBallotBox which implement the
VoteTallyer interface.


Speed

When compiled with go1.1 CloudForest achieves running times similar to implementations in
other languages. Using gccgo (4.8.0 at least) results in longer running times and is not
recommended until full go1.1 support is implemented in gcc 4.8.1.


Growforest Utility

"growforest" trains a forest using the following parameters which can be listed with -h

	Usage of growforest:
	  -contrastall=false: Include a shuffled artificial contrast copy of every feature.
	  -cost="": For categorical targets, a json string to float map of the cost of falsely identifying each category.
	  -cpuprofile="": write cpu profile to file
	  -entropy=false: Use entropy minimizing classification (target must be categorical).
	  -importance="": File name to output importance.
	  -impute=false: Impute missing values to feature mean/mode instead of filtering them out when splitting.
	  -l1=false: Use l1 norm regression (target must be numeric).
	  -leafSize=0: The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.
	  -mTry=0: Number of candidate features for each split. Inferred to ceil(swrt(nFeatures)) if <=0.
	  -nContrasts=0: The number of randomized artificial contrast features to include in the feature matrix.
	  -nCores=1: The number of cores to use.
	  -nSamples=0: The number of cases to sample (with replacement) for each tree grow. If <=0 set to total number of cases
	  -nTrees=100: Number of trees to grow in the predictor.
	  -rfpred="rface.sf": File name to output predictor forest in sf format.
	  -splitmissing=false: Split missing values onto a third branch at each node (experimental).
	  -target="": The row header of the target in the feature matrix.
	  -train="featurematrix.afm": AFM formated feature matrix containing training data.





Applyforrest Utility

"applyforest" applies a forest to the specified feature matrix and outputs predictions as a two column
(caselabel	predictedvalue) tsv.

	Usage of applyforest:
	  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
	  -preds="predictions.tsv": The name of a file to write the predictions into.
	  -rfpred="rface.sf": A predictor forest.



Errorrate Utility

errorrate calculates the error of a forest vs a testing data set and reports it to standard out

	Usage of errorrate:
	  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
	  -rfpred="rface.sf": A predictor forest.


Leafcount Utility

leafcount outputs counts of case case co-occurrence on leaf nodes (Brieman's proximity) and counts of the
number of times a feature is used to split a node containing each case (a measure of relative/local
importance).

	Usage of leafcount:
	  -branches="branches.tsv": a case by feature sparse matrix of leaf co-occurance in tsv format
	  -fm="featurematrix.afm": AFM formated feature matrix to use.
	  -leaves="leaves.tsv": a case by case sparse matrix of leaf co-occurance in tsv format
	  -rfpred="rface.sf": A predictor forest.


Feature Matrix Files

CloudForest borrows the annotated feature matrix (.afm) and stoicastic forest (.sf) file formats
from Timo Erkkila's rf-ace which can be found at https://code.google.com/p/rf-ace/

An annotated feature matrix (.afm) file is a tab delineated file with column and row headers. Columns represent cases and rows
represent features. A row header/feature id includes a prefix to specify the feature type

	"N:" Prefix for numerical feature id.
	"C:" Prefix for categorical feature id.
	"B:" Prefix for boolean feature id.

Categorical and boolean features use strings for their category labels. Missing values are represented
by "?","nan","na", or "null" (case insensitive). A short example:

	featureid	case1	case2	case3
	N:NumF1	0.0	.1	na
	C:CatF2 red	red	green


Stochastic Forest Files

A stochastic forest (.sf) file contains a forest of decision trees. The main advantage of this
format as opposed to an established format like json is that an sf file can be written iteratively
tree by tree and multiple .sf files can be combined with minimal logic required allowing for
massively parallel growth of forests with low memory use.

An .sf file consists of lines each of which is a comma separated list of key value pairs. Lines can
designate either a FOREST, TREE, or NODE. Each tree belongs to the preceding forest and each node to
the preceding tree. Nodes must be written in order of increasing depth.

CloudForest generates fewer fields then rf-ace but requires the following. Other fields will be
ignored

Forest requires forest type (only RF currently), target and ntrees:

	FOREST=RF|GBT|..,TARGET="$feature_id",NTREES=int

Tree requires only an int and the value is  ignored though the line is needed to designate a new tree:

	TREE=int

Node requires a path encoded so that the root node is specified by "*" and each split left or right as "L" or "R".
Leaf nodes should also define PRED such as "PRED=1.5" or "PRED=red". Splitter nodes should define SPLITTER with
a feature id inside of double quotes, SPLITTERTYPE=[CATEGORICAL|NUMERICAL] and a LVALUE term which can be either
a float inside of double quotes representing the highest value sent left or a ":" separated list of categorical
values sent left.

	NODE=$path,PRED=[float|string],SPLITTER="$feature_id",SPLITTERTYPE=[CATEGORICAL|NUMERICAL] LVALUES="[float|: separated list"

An example .sf file:

	FOREST=RF,TARGET="N:CLIN:TermCategory:NB::::",NTREES=12800
	TREE=0
	NODE=*,PRED=3.48283,SPLITTER="B:SURV:Family_Thyroid:F::::maternal",SPLITTERTYPE=CATEGORICAL,LVALUES="false"
	NODE=*L,PRED=3.75
	NODE=*R,PRED=1

Cloud forest can parse and apply .sf files generated by at least some versions of rf-ace.


References

The idea for (and trademark of the term) Random Forests originated with Leo Brieman and
Adele Cuttler. Their code and paper's can be found at:

http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

All code in CloudForest is original but some ideas for methods and optimizations were inspired by
Timo Erkilla's rf-ace and Andy Liaw and Matthew Wiener randomForest R package based on Brieman and
Cuttler's code:

https://code.google.com/p/rf-ace/
http://cran.r-project.org/web/packages/randomForest/index.html

The idea for Artificial Contrasts was found in:
Eugene Tuv, Alexander Borisov, George Runger and Kari Torkkola's paper "Feature Selection with
Ensembles, Artificial Variables, and Redundancy Elimination"
http://www.researchgate.net/publication/220320233_Feature_Selection_with_Ensembles_Artificial_Variables_and_Redundancy_Elimination/file/d912f5058a153a8b35.pdf

The idea for growing trees to minimize categorical entropy comes from Ross Quinlan's ID3:
http://en.wikipedia.org/wiki/ID3_algorithm

"The Elements of Statistical Learning" 2nd edition by Trevor Hastie, Robert Tibshirani and Jerome Friedman
was also consulted during development.

*/
package CloudForest
