python sklrf.py covtype.libsvm 
growforest -train covtype.libsvm -target "0" -nTrees 50 -nCores 8 -mTry 7
growforest -train covtypeNorm.arff -target "class" -nTrees 50 -nCores 8 -mTry 7
#time rf-ace --trainData covtype.fm --target C:class --nodeSize 1 -n 50 -e 8 -m 7