#Classifier for ssk

# import reuters data
from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def classifier_ssk (fm_train_dna=traindat,fm_test_dna=testdat,
		label_train_dna=label_traindat,C=1,maxlen=1,decay=1):
	from shogun import StringCharFeatures, BinaryLabels
	from shogun import LibSVM, SubsequenceStringKernel, DNA
	from shogun import ErrorRateMeasure

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	labels=BinaryLabels(label_train_dna)
	kernel=SubsequenceStringKernel(feats_train, feats_train, maxlen, decay);

	svm=LibSVM(C, kernel, labels);
	svm.train();

	out=svm.apply(feats_train);
	evaluator = ErrorRateMeasure()
	trainerr = evaluator.evaluate(out,labels)
	# print(trainerr)

	kernel.init(feats_train, feats_test)
	predicted_labels=svm.apply(feats_test).get_labels()
	# print predicted_labels

	return predicted_labels
