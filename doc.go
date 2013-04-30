/*
Package CloudForest implements decision trees for machine learning and includes a pure go
(golang) implementation of Breiman and Cutler's Random Forest for clasiffication and regression on
heterogenous numerical/catagorical data with missing values.

CloudForest is being developed in the Shumelivich Lab at the Institute
for Systems Biology and is released under a modified BSD style license.

Code and the Bug tracker can be found at https://github.com/ryanbressler/CloudForest


Caveats

When compiled with the default go 1.1 tool chain CloudForest achieves running times similar or
better then implementations in other languages. Using gccgo (4.8.0 at least) results in longer
running times and is not recomended at this time.

CloudForest is especially fast with data that includes lots of binary or low n catagorical data (ie
genomic variants) as it includeds optimized code to find the best spliter in these cases.


Goals and Quirks

CloudForest is intended to provide fast, comprehensible building blocks that can be used
to implement ensembels of decision trees. CloudForest is written in (somewhat) idomatic
go to allow a programer to implement and scale new models and analysis quickly instead
of having to modify complex code.

Go's support for first class functions is key to CloudForests flexiblity. For example
growing a decision tree using Breiman and Cutler's method can be done in an anonymous
function/closure passed to a tree's root node's Recurse method:

	tree.Root.Recurse(func(n *Node, innercases []int) {

		if (2 * leafSize) <= len(innercases) {
			SampleFirstN(&canidates, mTry)
			best, impDec := target.BestSplitter(fm, innercases, canidates[:mTry], itter, l, r)
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
to tabulate votes or extract scores structure.

Go's slices are used extensivelly in CloudForest to make it simple to interact with optimized code.
Many previous imlementations of Random Forest have avoided reallocation by reordering data in
place and keeping track of start and end indexes. In go, slices pointing at the same underlying
arrays make this sort of optimization transparent. For example a function like:

	func(s *Splitter) SplitInPlace(fm *FeatureMatrix, cases []int) (l []int, r []int)

can return left and right slices that point to the same underlying array as the origional
slice of cases but these slices should not have their values changed.

Code that is called repeatedly during training/tree growth also accepts pointers to slices that
will be reset to zero length and reused without reallocation. These slices won't contain meaningfull
data after the search is done but provide signifigant speed gains. Their use can be seen in
the l and r parmaters passed to BestSplitter in the the tree growing code above and functions
that accept them include:

	func (target *Feature) BestSplitter(fm *FeatureMatrix,
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

Future goals include a full set of comand line utilites and the implementation of various
measurs of importance, proximity and forest structure and related ensembel methods inluding
extra random trees and gradiant boosting trees.

Internally we will move towards the abstraction of a data feature to an interface to allow
extension to non catacagroical and numeric features and the further abstraction of collection
objects to  better support paralelization across many machines.


Overview

In CloudForest data is stored using the FeatureMatrix struct which contains Features.

The Feature struct  implments storage and methods for both catagorical and numerical data and is
responsible for calculations of impurity etc and the search for the best split.

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



A stoichastic forest (.sf) file contains a forest of decision trees. The main advantage of this
format as opposed to an established format like json is that an sf file can be written iterativelly
tree by tree and multiple .sf files can be combined with minimal logic required.

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

Cloud forest can parse and apply .sf files generated by at least some versions of rf-ace


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
