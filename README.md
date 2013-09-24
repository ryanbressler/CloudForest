CloudForest
==============

CloudForest implements fast, flexible ensembles of decision trees for machine
learning in pure Go (golang to search engines). It allows for a number of related algorithms
for classification, regression, feature selection and structure analysis on heterogeneous
numerical / categorical data with missing values. These include:

* Breiman and Cutler's Random Forest for Classification and Regression
* Adaptive Boosting (AdaBoost) Classification and Regression
* Gradient Boosting Tree Regression
* Entropy and Cost driven classification
* L1 regression
* Feature selection with artificial contrasts
* Proximity and model structure analysis
* Methods for Classification on Unbalanced Data

CloudForest has been optimized to minimize memory use, allow multi-core and learning with 
minimal allocations and perform especially well learning from categorical features 
with a small number of class labels. This includes binary data and genomic variant data which
may have class labels like "reference", "heterozygous", "homozygous".

File formats have been chosen to allow multi machine parallel learning. 

Command line utilities to grow, apply and analyze forests are provided in sub directories
or CloudForest can be used as a library.

This Document covers command line usage, file formats and some algorithmic background.

Documentation for coding against CloudForest has been generated with godoc and can be viewed live at:
http://godoc.org/github.com/ryanbressler/CloudForest

Pull requests and bug reports are welcome; Code Repo and Issue tracker can be found at:
https://github.com/ryanbressler/CloudForest

CloudForest is being developed in the Shumelivich Lab at the Institute for Systems
Biology.

Installation
-------------
With go and go path set up:

```bash
go get github.com/ryanbressler/CloudForest
go install github.com/ryanbressler/CloudForest/growforest
go install github.com/ryanbressler/CloudForest/applyforest
go install github.com/ryanbressler/CloudForest/leafcount
```

Quick Start
-------------

```bash
#grow a predictor forest with default parameters and save it to forest.sf
growforest -train train.fm -rfpred forest.sf -target B:FeatureName

#grow a 1000 tree forest using, 16 cores and report out of bag error with minimum leafSize 8 
growforest -train train.fm -rfpred forest.sf -target B:FeatureName -oob -nThreads 16 -nTrees 1000 -leafSize 8

#grow a 1000 tree forest evaluating half the features as candidates at each split and reporting 
#out of bag error after each tree to watch for convergence
growforest -train train.fm -rfpred forest.sf -target B:FeatureName -mTry .5 -progress 

#growforest with weighted random forest
growforest -train train.fm -rfpred forest.sf -target B:FeatureName -rfweights '{"true":2,"false":0.5}'

#report all growforest options
growforest -h

#Print the (balanced for classification, least squares for regression error rate on test data to standard out
applyforest -fm test.fm -rfpred forest.sf

#Apply the forest, report errorrate and save predictions
#Predictions are output in a tsv as:
#CaseLabel	Predicted	Actual
applyforest -fm test.fm -rfpred forest.sf -preds predictions.tsv

#Calculate counts of case vs case (leaves) and case vs feature (branches) proximity.
#Leaves are reported as:
#Case1 Case2 Count
#Branches Are Reported as:
#Case Feature Count
leafcount -train train.fm -rfpred forest.sf -leaves leaves.tsv -branches branches.tsv
```

Growforest Utility
--------------------

growforest trains a forest using the following parameters which can be listed with -h

Parameter's are implemented using go's parameter parser so that boolean parameters can be
set to true with a simple flag:
    
    #the following are equivalent
    growforest -oob
    growforest -oob=true

And equals signs and quotes are optional for other parameters:
	
    #the following are equivalent
	growforest -train featurematrix.afm
	growforest -train="featurematrix.afm"


 Basic options

 ```
   -target="": The row header of the target in the feature matrix.
   -train="featurematrix.afm": AFM formated feature matrix containing training data.
   -rfpred="rface.sf": File name to output predictor forest in sf format.
   -leafSize="0": The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.
   -mTry="0": Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.
   -nSamples="0": The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.
   -nTrees=100: Number of trees to grow in the predictor.
  
   -importance="": File name to output importance.
 
   -oob=false: Calculate and report oob error.
  
 ```

 Advanced Options

 ```
   -blacklist="": A list of feature id's to exclude from the set of predictors.
   -includeRE="": Filter features that DON'T match this RE.
   -blockRE="": A regular expression to identify features that should be filtered out.
   -impute=false: Impute missing values to feature mean/mode before growth.
   -nCores=1: The number of cores to use.
   -progress=false: Report tree number and running oob error.
   -oobpreds="": Calculate and report oob predictions in the file specified.
   -cpuprofile="": write cpu profile to file
   -multiboost=false: Allow multithreaded boosting which msy have unexpected results. (highly experimental)
   -nobag=false: Don't bag samples for each tree.
   -splitmissing=false: Split missing values onto a third branch at each node (experimental).
 ```

 Regression Options

 ```
   -gbt=0: Use gradient boosting with the specified learning rate.
   -l1=false: Use l1 norm regression (target must be numeric).
   -ordinal=false: Use ordinal regression (target must be numeric).
   -adaboost=false: Use Adaptive boosting (highly experimental for regression).
 ```

 Classification Options

 ```
   -adaboost=false: Use Adaptive boosting for classification.
   -balance=false: Balance bagging of samples by target class for unbalanced classification.
   -cost="": For categorical targets, a json string to float map of the cost of falsely identifying each category.
   -entropy=false: Use entropy minimizing classification (target must be categorical).
   -rfweights="": For categorical targets, a json string to float map of the weights to use for each catagory in Weighted RF.
 ```

Note: rfweights and cost should use json to specify the weights and or costs per class using the strings used to represent the class in the boolean or catagorical feature:

```
   growforest -rfweights '{"true":2,"false":0.5}'
```
 Randomizing Data

 Randomizing shuffling parts of the data or including shuffled "Artifichal Contrasts" can be useful to establish baselines for comparison.

 ```
   -permutate=false: Permutate the target feature (to establish random predictive power).
   -contrastall=false: Include a shuffled artificial contrast copy of every feature.
   -nContrasts=0: The number of randomized artificial contrast features to include in the feature matrix.
   -shuffleRE="": A regular expression to identify features that should be shuffled.
 ```




Applyforrest Utility
----------------------

applyforest applies a forest to the specified feature matrix and outputs predictions as a two column
(caselabel	predictedvalue) tsv.

```
Usage of applyforest:
  -fm="featurematrix.afm": AFM formated feature matrix containing data.
  -mean=false: Force numeric (mean) voteing.
  -mode=false: Force catagorical (mode) voteing.
  -preds="": The name of a file to write the predictions into.
  -rfpred="rface.sf": A predictor forest.
```

Leafcount Utility
-------------------

leafcount outputs counts of case case co-occurrence on leaf nodes (leaves.tsv, Brieman's proximity) and counts of the
number of times a feature is used to split a node containing each case (branches.tsv a measure of relative/local
importance).

```
Usage of leafcount:
  -branches="branches.tsv": a case by feature sparse matrix of leaf co-occurrence in tsv format
  -fm="featurematrix.afm": AFM formated feature matrix to use.
  -leaves="leaves.tsv": a case by case sparse matrix of leaf co-occurrence in tsv format
  -rfpred="rface.sf": A predictor forest.
```

DEPRECIATED Errorrate Utility (equivlent to applyforest with no -preds option)
-------------------

errorrate calculates the error of a forest vs a testing data set and reports it to standard out

```
Usage of errorrate:
  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
  -rfpred="rface.sf": A predictor forest.

```

Importance and Contrasts
--------------------------

Variable Importance in CloudForest is based on the as the mean decrease in impurity over all of
the splits made using a feature. It is output in a tsv as:

Feature DecreasePerUse UseCount DecresePerTree DecresePerTreeUsed TreeUsedCount MeanMinimalDepth

Where DecresePerTree is calculated over all trees, not just the ones the feature was used in and DecresePerTree.

Each of these scores has different properties:
* Per-use and per-tree-used scores may be more resistant to feature redundancy, 
* Per-tree-used and per-tree scores may better pick out complex effects.
* Mean Minimal Depth has been proposed (see "Random Survival Forests") as an alternative importance.

To provide a baseline for evaluating importance, artificial contrast features can be used by
including shuffled copies of existing features (-nContrasts, -contrastAll).

A feature that performs well when randomized (or when the target has been randomized) may be causing
overfitting. 

The option to permutate the target (-permutate) will establish a minimum random baseline. Using a 
regular expression (-shuffleRE) to shuffle part of the data can be useful in teasing out the contributions of 
different subsets of features. 


Feature Matrix Files
----------------------

CloudForest borrows the annotated feature matrix (.afm) and stochastic forest (.sf) file formats
from Timo Erkkila's rf-ace which can be found at https://code.google.com/p/rf-ace/

An annotated feature matrix (.afm) file is a tab delineated file with column and row headers. Columns represent cases and rows
represent features. A row header / feature id includes a prefix to specify the feature type

```
"N:" Prefix for numerical feature id.
"C:" Prefix for categorical feature id.
"B:" Prefix for boolean feature id.
```

Categorical and boolean features use strings for their category labels. Missing values are represented
by "?","nan","na", or "null" (case insensitive). A short example:

```
featureid	case1	case2	case3
N:NumF1	0.0	.1	na
C:CatF2 red	red	green
```


Stochastic Forest Files
-------------------------

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

Compiling for Speed
----------------------

When compiled with go1.1 CloudForest achieves running times similar to implementations in
other languages. Using gccgo (4.8.0 at least) results in longer running times and is not
recommended. This may change as gcc go addopts the go 1.1 way of implementing closures. 


References
-------------

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

Methods for classification from unbalanced data are covered in several papers:
http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf
http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3163175/
http://www.biomedcentral.com/1471-2105/11/523
http://bib.oxfordjournals.org/content/early/2012/03/08/bib.bbs006
http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0067863

    
