CloudForest
==============

CloudForest implements fast, flexible ensembles of decision trees for machine
learning in pure Go (golang). It includes implementations of Breiman
and Cutler's Random Forest for classification and regression on heterogeneous
numerical/categorical data with missing values and several related algorithms
including entropy and cost driven classification, L1 regression and feature
selection with artificial contrasts. It is intended to allow algorithms
to be quickly modified for your needs.

Command line utilities to grow, apply and canalize forests are provided in sub directories
or CloudForest can be used as a library.

This Document covers comand line ussage and provides algorythmic background.

Documentation has been generated with godoc and can be viewed live at:
http://godoc.org/github.com/ryanbressler/CloudForest

Pull requests and bug reports are welcome; Code Repo and Issue tracker can be found at:
https://github.com/ryanbressler/CloudForest

CloudForest is being developed in the Shumelivich Lab at the Institute for Systems
Biology.

Instalation
-------------
With go and go path set up:

```bash
go get github.com/ryanbressler/CloudForest
go install github.com/ryanbressler/CloudForest/growforest
go install github.com/ryanbressler/CloudForest/errorrate
go install github.com/ryanbressler/CloudForest/applyforest
go install github.com/ryanbressler/CloudForest/leafcount
```

Quick Start
-------------

```bash
#grow a predictor forest with default pamaters and save it to forest.sf
#run growforest -h for more options
growforest -train train.fm -rfpred forest.sf -target B:FeatureName

#grow a 1000 tree forest using, 16 cores and report out of bag error
growforest -train train.fm -rfpred forest.sf -target B:FeatureName -oob -nThreads 16 -nTrees 1000

#grow a 1000 tree forest evaluating half the features as canidates at each split and reporting oob error
#after each tree to watch for convergence
growforest -train train.fm -rfpred forest.sf -target B:FeatureName -mTry .5 -progress


#Print the (balanced for classification, least squares for regression error rate on test data to standard out
errorrate -fm test.fm -rfpred forest.sf

#Apply the forest, report errorrate and save predictions
#Predictions are output in a tsv as:
#CaseLabel	Predicted	Actual
errorrate -fm test.fm -rfpred forest.sf -preds predictions.tsv

#Calculate counts of case vs case (leaves) and case vs feature (branches) proximity.
#Leaves are reported as:
#Case1 Case2 Count
#Branches Are Reported as:
#Case Feature Count
leafcount -train train.fm -rfpred forest.sf -leaves leaves.tsv -brances branches.tsv
```

Growforest Utility
------------------

"growforest" trains a forest using the following parameters which can be listed with -h

```
Usage of growforest:
  -adaboost=false: Use Adaptive boosting for regresion/classification.
  -balance=false: Ballance bagging of samples by target class for unbalanced classification.
  -blacklist="": A list of feature id's to exclude from the set of predictors.
  -contrastall=false: Include a shuffled artificial contrast copy of every feature.
  -cost="": For categorical targets, a json string to float map of the cost of falsely identifying each category.
  -cpuprofile="": write cpu profile to file
  -entropy=false: Use entropy minimizing classification (target must be categorical).
  -gbt=0: Use gradiant boosting with the specified learning rate.
  -importance="": File name to output importance.
  -impute=false: Impute missing values to feature mean/mode before growth.
  -l1=false: Use l1 norm regression (target must be numeric).
  -leafSize="0": The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.
  -mTry="0": Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.
  -multiboost=false: Allow multithreaded boosting which msy have unexpected results. (highly experimental)
  -nContrasts=0: The number of randomized artificial contrast features to include in the feature matrix.
  -nCores=1: The number of cores to use.
  -nSamples="0": The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.
  -nTrees=100: Number of trees to grow in the predictor.
  -nobag=false: Don't bag samples for each tree.
  -oob=false: Calculte and report oob error.
  -oobpreds="": Calculate and report oob predictions in the file specified.
  -ordinal=false: Use ordinal regression (target must be numeric).
  -permutate=false: Permutate the target feature (to establish random predictive power).
  -progress=false: Report tree number and running oob error.
  -rfpred="rface.sf": File name to output predictor forest in sf format.
  -shuffleRE="": A regular expression to identify features that should be shuffled.
  -splitmissing=false: Split missing values onto a third branch at each node (experimental).
  -target="": The row header of the target in the feature matrix.
  -train="featurematrix.afm": AFM formated feature matrix containing training data.
 ```





Applyforrest Utility
--------------------

"applyforest" applies a forest to the specified feature matrix and outputs predictions as a two column
(caselabel	predictedvalue) tsv.

```
Usage of applyforest:
  -fm="featurematrix.afm": AFM formated feature matrix containing data.
  -mean=false: Force numeric (mean) voteing.
  -mode=false: Force catagorical (mode) voteing.
  -preds="predictions.tsv": The name of a file to write the predictions into.
  -rfpred="rface.sf": A predictor forest.
```



Errorrate Utility
-----------------

errorrate calculates the error of a forest vs a testing data set and reports it to standard out

```
Usage of errorrate:
  -fm="featurematrix.afm": AFM formated feature matrix containing test data.
  -rfpred="rface.sf": A predictor forest.

```


Leafcount Utility
-----------------

leafcount outputs counts of case case co-occurrence on leaf nodes (Brieman's proximity) and counts of the
number of times a feature is used to split a node containing each case (a measure of relative/local
importance).

```
Usage of leafcount:
  -branches="branches.tsv": a case by feature sparse matrix of leaf co-occurrence in tsv format
  -fm="featurematrix.afm": AFM formated feature matrix to use.
  -leaves="leaves.tsv": a case by case sparse matrix of leaf co-occurrence in tsv format
  -rfpred="rface.sf": A predictor forest.
```


Feature Matrix Files
--------------------

CloudForest borrows the annotated feature matrix (.afm) and stoicastic forest (.sf) file formats
from Timo Erkkila's rf-ace which can be found at https://code.google.com/p/rf-ace/

An annotated feature matrix (.afm) file is a tab delineated file with column and row headers. Columns represent cases and rows
represent features. A row header/feature id includes a prefix to specify the feature type

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
-----------------------

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

    
