Benchmarks on Forest Coverage Data
================================================

Learner | CloudForest | scikit.learn 0.14.1 | scikit.learn 0.15 | CloudForest 
--------|-------------|---------------------|-------------------|------------
Format  | libsvm      | libsvm              | libsvm            | arff
Time    | 38 seconds  | ??? seconds         | 30 seconds        | 29 seconds

The arff format records which variables are binary or catagorical allowing cloudforest to use appropriate splitters for greater speed. Scikit.learn treats all data as numerical. This data set was chosen to allow comparison with benchmarks by [wise.io](http://about.wise.io/blog/2013/07/15/benchmarking-random-forest-part-1/) and [Alex Rubinsteyn](http://blog.explainmydata.com/2014/03/big-speedup-for-random-forest-learning.html). 

Cloudforest and scikit.learn 0.15 were checked out on 3/10/2014. 

Hardware
---------
Benchmarks were performed using 8 hyperthreads on a 15-inc MacBook Pro 10,1 with a 2.4 Ghz Intel Core i7 (I7-3635QM) with per Core L3 cahe of 256 KB, 6 MB of L3 cache and 8Gb of 1600 MHz ram.

Data Sources
------------

The forest coverage data set in libsvm format  was aquired [from here](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype).

Scaled forest coverage in arff format with catagorical variables not converted to numerical was aquired [from the MOA project](http://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/).