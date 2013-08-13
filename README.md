CloudForest
============

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
===========
With go and go path set up:

'''bash
go get github.com/ryanbressler/CloudForest
go install github.com/ryanbressler/CloudForest/growforest
go install github.com/ryanbressler/CloudForest/errorrate
go install github.com/ryanbressler/CloudForest/applyforest
go install github.com/ryanbressler/CloudForest/leafcount
'''

Quick Start
===========
'''bash
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
'''

Advanced Ussage
===============


    
