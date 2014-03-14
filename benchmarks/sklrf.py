import sys
from sklearn.datasets import load_svmlight_file

from sklearn.ensemble import RandomForestClassifier

from time import time

import numpy as np

# def main():
fn = sys.argv[1]
X,Y = load_svmlight_file(fn)

rf_parameters = {
	"n_estimators": 50,
	"n_jobs": 8,
	"max_features":7,
	"bootstrap":True,
	"verbose":1
}
clf = RandomForestClassifier(**rf_parameters)
X = X.toarray()

print clf

t0 = time()
clf.fit(X, Y)
train_time = time() - t0
print "Training on %s took %s"%(fn, train_time)

score = clf.score(X, Y)
count = np.sum(clf.predict(X)==Y)
print "Score: %s, %s / %s "%(score, count, len(Y))

# if __name__ == '__main__':
# 	main()
 	