/*
Package CloudForest implements ensembles of decision trees for machine
learning in pure Go (golang to search engines). It allows for a number of related algorithms
for classification, regression, feature selection and structure analysis on heterogeneous
numerical/categorical data with missing values. These include:

	* Breiman and Cutler's Random Forest for Classification and Regression

	* Adaptive Boosting (AdaBoost) Classification

	* Gradiant Boosting Tree Regression

	* Entropy and Cost driven classification

	* L1 regression

	* Feature selection with artificial contrasts

	* Proximity and model structure analysis

	* Roughly balanced bagging for unbalanced classification

The API hasn't stabilized yet and may change rapidly. Tests and benchmarks have been performed
only on embargoed data sets and can not yet be released.

Library Documentation is in code and can be viewed with godoc or live at:
http://godoc.org/github.com/ryanbressler/CloudForest

Documentation of command line utilities and file formats can be found in README.md, which can be
viewed fromated on github:
http://github.com/ryanbressler/CloudForest

Pull requests and bug reports are welcome.

CloudForest was created by Ryan Bressler and is being developed in the Shumelivich Lab at
the Institute for Systems Biology for use on genomic/biomedical data with partial support
from The Cancer Genome Atlas and the Inova Translational Medicine Institute.


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
			best, impDec := fm.BestSplitter(target, innercases, candidates[:mTry], false, allocs)
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


Stackable Interfaces

Decision tree's are grown with the goal of reducing "Impurity" which is usually defined as Gini
Impurity for categorical targets or mean squared error for numerical targets. CloudForest grows
trees against the Target interface which allows for alternative definitions of impurity. CloudForest
includes several alternative targets:

 EntropyTarget : For use in entropy minimizing classification
 RegretTarget  : For use in classification driven by differing costs in mis-categorization.
 L1Target      : For use in L1 norm error regression (which may be less sensitive to outliers).
 OrdinalTarget : For ordinal regression

Additional targets can be stacked on top of these target to add boosting functionality:
 GradBoostTarget : For Gradient Boosting Regression
 AdaBoostTarget  : For Adaptive Boosting Classification



Efficient Splitting

Repeatedly splitting the data and searching for the best split at each node of a decision tree
are the most computationally intensive parts of decision tree learning and CloudForest includes
optimized code to perform these tasks.

Go's slices are used extensively in CloudForest to make it simple to interact with optimized code.
Many previous implementations of Random Forest have avoided reallocation by reordering data in
place and keeping track of start and end indexes. In go, slices pointing at the same underlying
arrays make this sort of optimization transparent. For example a function like:

	func(s *Splitter) SplitInPlace(fm *FeatureMatrix, cases []int) (l []int, r []int)

can return left and right slices that point to the same underlying array as the original
slice of cases but these slices should not have their values changed.

Functions used while searching for the best split also accepts pointers to reusable slices and
structs to maximize speed by keeping memory allocations to a minimum. BestSplitAllocs contains
pointers to these items and its use can be seen in functions like:

	func (fm *FeatureMatrix) BestSplitter(target Target,
		cases []int,
		candidates []int,
		extraRandom bool,
		allocs *BestSplitAllocs) (s *Splitter, impurityDecrease float64)

	func (f *Feature) BestSplit(target Target,
		cases *[]int,
		parentImp float64,
		randomSplit bool,
		allocs *BestSplitAllocs) (bestNum float64, bestCat int, bestBigCat *big.Int, impurityDecrease float64)


For categorical predictors, BestSplit will also attempt to intelligently choose between 4
different implementations depending on user input and the number of categories.
These include exhaustive, random, and iterative searches for the best combination of categories
implemented with bitwise operations against int and big.Int. See BestCatSplit, BestCatSplitIter,
BestCatSplitBig and BestCatSplitIterBig.

All numerical predictors are handled by BestNumSplit which
relies on go's sorting package.

Parallelism and Scaling

Training a Random forest is an inherently parallel process and CloudForest is designed
to allow parallel implementations that can tackle large problems while keeping memory
usage low by writing and using data structures directly to/from disk.

Trees can be grown in separate go routines. The growforest utility provides an example
of this that uses go routines and channels to grow trees in parallel and write trees
to disk as the are finished by the "worker" go routines. The few summary statistics
like mean impurity decrease per feature (importance) can be calculated using thread
safe data structures like RunningMean.

Trees can also be grown on separate machines. The .sf stochastic forest format
allows several small forests to be combined by concatenation and the ForestReader
and ForestWriter structs allow these forests to be accessed tree by tree (or even node
by node) from disk.

For data sets that are too big to fit in memory on a single machine Tree.Grow and
FeatureMatrix.BestSplitter can be reimplemented to load candidate features from disk,
distributed database etc.


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



*/
package CloudForest
