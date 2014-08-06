import uuid
import pandas as pd
import subprocess
import numpy as np

__doc__="""The CFClassifier module includes the CFClassifier class which will wrap
calls to cloudforests growforest and applyforest utilities to be called as a scikit-learn
classifier.

It works via writting uuid identified temp files to disk in the current working directory so
it has more overhead then a pure in memory implementation but handle problems where the 
model is too large to fit in system memory.
"""

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def writearff(fo, df, target="", unique=[]):
	"""writearff writes a pandasdataframe, df, to a file like object fo"""

	fo.write("@RELATION %(target)s\n\n"%{"target":target})

	#print df[target]

	for col in df.columns:
		#print df[col].dtype
		coltype="NUMERIC"

		if df.dtypes[col] == bool:
			coltype = "{True,False}"

		if target!="" and col == target:
			coltype = "{%(values)s}"%{"values":",".join([str(v) for v in unique])}
		

		

		fo.write("@ATTRIBUTE %(name)s %(type)s\n"%{"name":col,"type":coltype})

	fo.write("\n@DATA\n")
	df.to_csv(fo, na_rep="NA", index=False, header=False)

class CFClassifier:
	"""CFClassifier wraps command line calls to cloudforest's growforest 
	and applyforest for use as a scikit-learn Classifier. It will write
	temporary files to your workding directory."""

	options = ""


	def __init__(self, optionstring):
		self.options = optionstring
		self.uuid = uuid.uuid1()
	
	def fit(self, X, y):
		df = pd.DataFrame(X).copy()
		target = "%(uuid)s.target"%{"uuid":self.uuid}
		fn = "%(uuid)s.train.cloudforest.arff"%{"uuid":self.uuid}
		self.forest = "%(uuid)s.forest.cloudforest.sf"%{"uuid":self.uuid}

		
		self.unique = np.unique(y)

		#print y
		df[target] = np.array(y,dtype=bool)
		#print df[target]
		
		
		fo = open(fn,"w")
		writearff(fo,df,target,self.unique)
		fo.close()

		invocation = "growforest -train %(data)s -target %(target)s -rfpred %(forest)s %(options)s"%{"data":fn,
			"target":target,
			"forest":self.forest,
			"options":self.options}

		#print invocation

		subprocess.call(invocation, shell=True)

	def predict(self, X):
		df = pd.DataFrame(X)
		fn = "%(uuid)s.test.cloudforest.arff"%{"uuid":self.uuid}
		preds = "%(uuid)s.preds.cloudforest.tsv"%{"uuid":self.uuid}
		
		fo = open(fn,"w")
		writearff(fo,df)
		fo.close()

		invocation = "applyforest -fm %(data)s -rfpred %(forest)s -preds %(preds)s"%{"data":fn,
			"forest":self.forest,
			"preds": preds}

		subprocess.call(invocation, shell=True)

		fo =open(preds)
		predictions = []
		for line in fo:
			vs= line.rstrip().split()
			predictions.append(vs[1])
		fo.close()

		return np.array(predictions,dtype=int)

	def predict_proba(self, X):
		df = pd.DataFrame(X)
		fn = "%(uuid)s.test.cloudforest.arff"%{"uuid":self.uuid}
		votes = "%(uuid)s.votes.cloudforest.tsv"%{"uuid":self.uuid}
		
		
		fo = open(fn,"w")
		writearff(fo,df)
		fo.close()

		invocation = "applyforest -fm %(data)s -rfpred %(forest)s -votes %(votes)s"%{"data":fn,
			"forest":self.forest,
			"votes":votes}



		subprocess.call(invocation, shell=True)

		fo =open(votes)

		header = 0
		votes = 0

		
		line = fo.next()
		vs = line.split()[1:]
		if vs[0]=="True" or vs[0]=="False":
			header = np.array([strtobool(v) for v in vs],dtype=bool)
			votes = np.loadtxt(fo, dtype="int")
		else:
			header = np.array([int(v) for v in vs],dtype=int)
			votes = np.loadtxt(fo, dtype="int")
		fo.close()

		vote_totals = np.sum(votes[:,1:],axis=1)

		#print vote_totals.shape, votes.shape, self.unique.shape, self.unique

		probs = []
		for v in self.unique:
			if v in header:
				probs.append(np.array(votes[:,1:][:,header==v],dtype=float).T/np.array(vote_totals,dtype=float)[0])
			else:
				probs.append(np.zeros_like(vote_totals))



		return np.dstack(probs)[0]

