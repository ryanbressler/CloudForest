package main

import "flag"

var (
	fm             string
	rf             string
	targetname     string
	imp            string
	costs          string
	dentropy       string
	adacosts       string
	rfweights      string
	blacklist      string
	nCores         int
	StringnSamples string
	StringmTry     string
	StringleafSize string
	maxDepth       int
	shuffleRE      string
	blockRE        string
	includeRE      string
	multiboost     bool
	nobag          bool
	balance        bool
	balanceby      string
	ordinal        bool
	permutate      bool
	dotest         bool
	testfm         string
	scikitforest   string
	noseed         bool
	unlabeled      string
	trans_alpha    float64
	trans_beta     float64
	nTrees         int
	ace            int
	cutoff         float64
	nContrasts     int
	contrastAll    bool
	impute         bool
	extra          bool
	splitmissing   bool
	l1             bool
	density        bool
	vet            bool
	positive       string
	cpuprofile     string
	NP             bool
	NP_pos         string
	NP_a           float64
	NP_k           float64
	evaloob        bool
	force          bool
	entropy        bool
	oob            bool
	jungle         bool
	caseoob        string
	progress       bool
	adaboost       bool
	hellinger      bool
	gradboost      float64
)

func init() {
	flag.StringVar(&fm, "train", "featurematrix.afm", "AFM formated feature matrix containing training data.")
	flag.StringVar(&rf, "rfpred", "", "File name to output predictor forest in sf format.")
	flag.StringVar(&targetname, "target", "", "The row header of the target in the feature matrix.")
	flag.StringVar(&imp, "importance", "", "File name to output importance.")
	flag.StringVar(&costs, "cost", "", "For categorical targets, a json string to float map of the cost of falsely identifying each category.")
	flag.StringVar(&dentropy, "dentropy", "", "Class disutilities for disutility entropy.")
	flag.StringVar(&adacosts, "adacost", "", "Json costs for cost sentive AdaBoost.")
	flag.StringVar(&rfweights, "rfweights", "", "For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.")
	flag.StringVar(&blacklist, "blacklist", "", "A list of feature id's to exclude from the set of predictors.")

	flag.IntVar(&nCores, "nCores", 1, "The number of cores to use.")

	flag.StringVar(&StringnSamples, "nSamples", "0", "The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.")

	flag.StringVar(&StringmTry, "mTry", "0", "Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.")

	flag.StringVar(&StringleafSize, "leafSize", "0", "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	flag.IntVar(&maxDepth, "maxDepth", 0, "Maximum tree depth. Ignored if 0.")

	flag.StringVar(&shuffleRE, "shuffleRE", "", "A regular expression to identify features that should be shuffled.")

	flag.StringVar(&blockRE, "blockRE", "", "A regular expression to identify features that should be filtered out.")

	flag.StringVar(&includeRE, "includeRE", "", "Filter features that DON'T match this RE.")

	flag.StringVar(&unlabeled, "trans_unlabeled", "", "Class to treat as unlabeled for transduction forests.")

	flag.Float64Var(&trans_alpha, "trans_alpha", 10.0, "Weight of unsupervised term in transduction impurity.")

	flag.Float64Var(&trans_beta, "trans_beta", 0.0, "Multiple to penalize unlabeled class by.")

	flag.IntVar(&nTrees, "nTrees", 100, "Number of trees to grow in the predictor.")

	flag.IntVar(&ace, "ace", 0, "Number ace permutations to do. Output ace style importance and p values.")
	flag.Float64Var(&cutoff, "cutoff", 0.0, "P-value cutoff to apply to features for last forest after ACE.")

	flag.IntVar(&nContrasts, "nContrasts", 0, "The number of randomized artificial contrast features to include in the feature matrix.")

	flag.String("cpuprofile", "", "write cpu profile to file")

	flag.BoolVar(&contrastAll, "contrastall", false, "Include a shuffled artificial contrast copy of every feature.")

	flag.BoolVar(&impute, "impute", false, "Impute missing values to feature mean/mode before growth.")

	flag.BoolVar(&extra, "extra", false, "Grow Extra Random Trees (supports learning from numerical variables only).")

	flag.BoolVar(&splitmissing, "splitmissing", false, "Split missing values onto a third branch at each node (experimental).")

	flag.BoolVar(&l1, "l1", false, "Use l1 norm regression (target must be numeric).")

	flag.BoolVar(&density, "density", false, "Build density estimating trees instead of classification/regression trees.")

	flag.BoolVar(&vet, "vet", false, "Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.")

	flag.StringVar(&positive, "positive", "True", "Positive class to output probabilities for.")

	flag.BoolVar(&NP, "NP", false, "Do approximate Neyman-Pearson classification.")

	flag.StringVar(&NP_pos, "NP_pos", "1", "Class label to constrain percision in NP classification.")

	flag.Float64Var(&NP_a, "NP_a", 0.1, "Constraint on percision in NP classification [0,1]")

	flag.Float64Var(&NP_k, "NP_k", 100, "Weight of constraint in NP classification [0,Inf+)")

	flag.BoolVar(&evaloob, "evaloob", false, "Evaluate potential splitting features on OOB cases after finding split value in bag.")

	flag.BoolVar(&force, "force", false, "Force at least one non constant feature to be tested for each split.")

	flag.BoolVar(&entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	flag.BoolVar(&oob, "oob", false, "Calculate and report oob error.")

	flag.BoolVar(&jungle, "jungle", false, "Grow unserializable and experimental decision jungle with node recombination.")

	flag.StringVar(&caseoob, "oobpreds", "", "Calculate and report oob predictions in the file specified.")

	flag.BoolVar(&progress, "progress", false, "Report tree number and running oob error.")

	flag.BoolVar(&adaboost, "adaboost", false, "Use Adaptive boosting for regression/classification.")

	flag.BoolVar(&hellinger, "hellinger", false, "Build trees using hellinger distance.")

	flag.Float64Var(&gradboost, "gbt", 0.0, "Use gradient boosting with the specified learning rate.")

	flag.BoolVar(&multiboost, "multiboost", false, "Allow multi-threaded boosting which may have unexpected results. (highly experimental)")

	flag.BoolVar(&nobag, "nobag", false, "Don't bag samples for each tree.")

	flag.BoolVar(&balance, "balance", false, "Balance bagging of samples by target class for unbalanced classification.")

	flag.StringVar(&balanceby, "balanceby", "", "Roughly balanced bag the target within each class of this feature.")

	flag.BoolVar(&ordinal, "ordinal", false, "Use ordinal regression (target must be numeric).")

	flag.BoolVar(&permutate, "permute", false, "Permute the target feature (to establish random predictive power).")

	flag.BoolVar(&dotest, "selftest", false, "Test the forest on the data and report accuracy.")

	flag.StringVar(&testfm, "test", "", "Data to test the model on.")

	flag.StringVar(&scikitforest, "scikitforest", "", "Write out a (partially complete) scikit style forest in json.")

	flag.BoolVar(&noseed, "noseed", false, "Don't seed the random number generator from time.")

}
