import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# N-GRAM KERNEL

#Compute kernel for documents s and t
def compute(s,t,n):
    tfidfVectorizer = TfidfVectorizer(analyzer = "char",
                                        tokenizer = None,
                                        preprocessor = None,
                                        ngram_range=(n, n))
    docs = tfidfVectorizer.fit_transform([s,t])
    docs = docs.toarray()
    return np.dot(docs[0], docs[1])/np.sqrt(np.dot(docs[0], docs[0])*np.dot(docs[1], docs[1]))
