from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

cachedStopWords = stopwords.words("english")
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords] #here we need to put spaces where the cachedStopWords
    tokens = (list(map(lambda token: PorterStemmer().stem(token),words)))
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
    return filtered_tokens

def represent(documents):
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    
    # Tokenisation
   # vectorizer = TfidfVectorizer(tokenizer=tokenize) # we don't need that
    
    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    # Transform multilabel labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id]) 
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])
    
    return (vectorised_train_documents, train_labels, vectorised_test_documents, test_labels)
    
def train_classifier(train_docs, train_labels):
    #use classifier ssk
    return classifier

def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
