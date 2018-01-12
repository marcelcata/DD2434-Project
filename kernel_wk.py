import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Bag-of-words kernel

#Compute kernel for documents s and t
def compute(s,t):
    tfidfVectorizer = TfidfVectorizer(analyzer = "word",
                                        tokenizer = None,
                                        preprocessor = None,
                                        stop_words = "english",
                                        smooth_idf=False,
                                        sublinear_tf=True)
    docs = tfidfVectorizer.fit_transform([s,t])
    docs = docs.toarray()
    return np.dot(docs[0], docs[1])/np.sqrt(np.dot(docs[0], docs[0])*np.dot(docs[1], docs[1]))
