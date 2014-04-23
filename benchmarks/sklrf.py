import sys
from sklearn.datasets import load_svmlight_file

from sklearn.ensemble import RandomForestClassifier

from time import time

import numpy as np


def dumptree(atree, fn):
	from sklearn import tree
	f = open(fn,"w")
	tree.export_graphviz(atree,out_file=f)
	f.close()

# def main():
fn = sys.argv[1]
X,Y = load_svmlight_file(fn)

rf_parameters = {
	"n_estimators": 2000,
	"n_jobs": 8
}
clf = RandomForestClassifier(**rf_parameters)
X = X.toarray()

print clf

print "Starting Training"
t0 = time()
clf.fit(X, Y)
train_time = time() - t0
print "Training on %s took %s"%(fn, train_time)

if len(sys.argv) == 2:
	score = clf.score(X, Y)
	count = np.sum(clf.predict(X)==Y)
	print "Score: %s, %s / %s "%(score, count, len(Y))
else:
	fn = sys.argv[2]
	X,Y = load_svmlight_file(fn)
	X = X.toarray()
	score = clf.score(X, Y)
	count = np.sum(clf.predict(X)==Y)
	c1 = np.sum(clf.predict(X[Y==1])==Y[Y==1] )
	c0 = np.sum(clf.predict(X[Y==0])==Y[Y==0] )
	l = len(Y)
	print "Testing Score: %s, %s / %s, %s, %s, %s "%(score, count, l,  c1, c0, (float(c1)/float(sum(Y==1))+float(c0)/float(sum(Y==0)))/2.0)


# if __name__ == '__main__':
# 	main()
 	