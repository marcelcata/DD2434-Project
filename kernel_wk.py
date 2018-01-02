import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer

class Stemmer(object):
    def __init__(self):
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc)]

#Compute kernel for documents s and t
def compute(s,t):
    tfidfVectorizer = TfidfVectorizer(analyzer = "word",
                                        tokenizer = Stemmer(),
                                        preprocessor = None,
                                        stop_words = "english",
                                        smooth_idf=False,
                                        sublinear_tf=True)
    docs = tfidfVectorizer.fit_transform([s,t])
    docs = docs.toarray()
    return np.dot(docs[0], docs[1])
