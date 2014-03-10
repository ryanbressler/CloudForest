python sklrf.py covtype.libsvm 
growforest -train ../benchmarks/covtype.libsvm -target "0" -nTrees 50 -nCores 8 -mTry 7
growforest -train ../benchmarks/covtypeNorm.arff -target "class" -nTrees 50 -nCores 8 -mTry 7