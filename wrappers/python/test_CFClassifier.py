import unittest
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from CFClassifier import CFClassifier
import os.path

import numpy as np

from sklearn.metrics import roc_auc_score

class TestCFClassifier(unittest.TestCase):

	def test_iris(self):
		"""Check consistency on dataset iris."""

		# also load the iris dataset
		# and randomly permute it
		iris = datasets.load_iris()
		rng = check_random_state(0)
		perm = rng.permutation(iris.target.size)
		iris.data = iris.data[perm]
		iris.target = iris.target[perm]

		

		clf = CFClassifier("")
		clf.fit(iris.data, iris.target)

		self.assertTrue(os.path.isfile(clf.forest))

		preds = clf.predict(iris.data)


		predicted_ratio = float(np.sum(preds==iris.target))/float(len(iris.target))
		print predicted_ratio

		self.assertGreaterEqual(predicted_ratio, .97) 

		probs = clf.predict_proba(iris.data)


		bin_idx=iris.target!=2

		roc_auc = roc_auc_score(iris.target[bin_idx], probs[bin_idx,1])

		self.assertGreaterEqual(roc_auc, .97) 




		#score = clf.score(iris.data, iris.target)

		#assert_greater(score, 0.9, "Failed with criterion %s and score = %f"
		                      # % (criterion, score)

if __name__ == '__main__':
    unittest.main()