from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support
import numpy as np
import kernel_wk
import kernel_ngk
import kernel_ssk
import _pickle as pickle
import string


cachedStopWords = stopwords.words("english")
ACQ = 'acq'
CORN = 'corn'
CRUDE = 'crude'
EARN = 'earn'
TRAIN = 'train'
TEST = 'test'
CATEGORIES = [ACQ, CORN, CRUDE, EARN]
SPLITS = [TRAIN, TEST]

LOAD_MATRICES = 0
SAVE_MATRICES = 1

SIZES = {
    TRAIN: {
        ACQ: 114,
        CORN: 38,
        CRUDE: 76,
        EARN: 152
    },
    TEST: {
        ACQ: 25,
        CORN: 10,
        CRUDE: 15,
        EARN: 40
    }
}


########################### PREPROCESSING METHODS
def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def initialize():
    filename = 'dataset/subset_470.pkl'
    dataset = load_dataset(filename)
    labels = {
        'acq': 0,
        'corn': 1,
        'crude': 2,
        'earn': 3
    }

    #select documents and labels
    train_docs = [tuple[0] for tuple in dataset[TRAIN]]
    train_labels = np.array([labels[tuple[1]] for tuple in dataset[TRAIN]])
    test_docs = [tuple[0] for tuple in dataset[TEST]]
    test_labels = np.array([labels[tuple[1]] for tuple in dataset[TEST]])
    return train_docs, train_labels, test_docs, test_labels

def preprocessing(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    for c in string.punctuation:
        text = text.replace(c,"")
    return text.lower()

def gram_matrices(kernel_type, train_docs, train_labels, test_docs, test_labels, k, m_lambda):

    n_train = len(train_docs)
    n_test = len(test_docs)
    train_matrix = np.zeros((n_train,n_train))
    test_matrix = np.zeros((n_test,n_train))

    print(n_train)
    print(n_test)

    for i in range(n_train):
        print(i)
        for j in range(i+1,n_train):
            if kernel_type == 'wk':
                train_matrix[i][j] = kernel_wk.compute(train_docs[i],train_docs[j])
            elif kernel_type == 'ngram':
                train_matrix[i][j] = kernel_ngk.compute(train_docs[i],train_docs[j], k)
            elif kernel_type == 'ssk':
                train_matrix[i][j] = kernel_ssk.compute(train_docs[i],train_docs[j], k, m_lambda)

    train_matrix = train_matrix + train_matrix.T + np.eye(n_train)

    for i in range(n_test):
        print(i)
        for j in range(n_train):
            if kernel_type == 'wk':
                test_matrix[i][j] = kernel_wk.compute(test_docs[i],train_docs[j])
            elif kernel_type == 'ngram':
                test_matrix[i][j] = kernel_ngk.compute(test_docs[i],train_docs[j], k)
            elif kernel_type == 'ssk':
                test_matrix[i][j] = kernel_ssk.compute(test_docs[i],train_docs[j], k, m_lambda)

    return train_matrix, train_labels, test_matrix, test_labels


########################### USING THE KERNELS
def extract_features(kernel_type, train_docs, train_labels, test_docs, test_labels):

    # Tokenisation
    if kernel_type == 'wk':
        vectorizer = TfidfVectorizer(tokenizer=kernel_wk.word_kernel)
    elif kernel_type == 'ngram':
        vectorizer = TfidfVectorizer(tokenizer=kernel_ngk) #not implemented yet
    elif kernel_type == 'ssk':
        vectorizer = TfidfVectorizer(tokenizer=kernel_ssk) #not implemented yet
    else:
        print('Kernel not found.')

    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    return (vectorised_train_documents, train_labels, vectorised_test_documents, test_labels)




############################ CLASSIFIER
def train_classifier(train_docs, train_labels):
   # kernel_wk.compute(train_docs, test_docs)
    #classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier = SVC(kernel='precomputed')
    classifier.fit(train_docs, train_labels) #instead of train_docs -> kernel_train
    #classifier.fit(kernel_train, train_labels) #instead of train_docs -> kernel_train
    return classifier


############################ EVALUATION FUNCTIONS
def evaluate_macro_micro(test_labels, predictions):
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

def evaluate_acq(test_labels, predictions):

    labels = [0]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("acq")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def evaluate_earn(test_labels, predictions):
    labels = [3]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("earn")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def evaluate_corn(test_labels, predictions):
    labels = [1]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("corn")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def evaluate_crude(test_labels, predictions):
    labels = [2]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("crude")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def evaluate(test_labels, predictions):
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predictions)
    print("eval")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))


# Prepare Dataset
train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw = initialize()
# Get the feature according to the chosen kernel
kernel_type = 'wk'
k = 2
m_lambda = 0.5
#train_docs, train_labels, test_docs, test_labels = extract_features(kernel_type,
#                                            train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw)

train_docs_raw = [preprocessing(doc) for doc in train_docs_raw]
test_docs_raw = [preprocessing(doc) for doc in test_docs_raw]

if LOAD_MATRICES:
    if kernel_type == 'wk':
        train_docs = np.load('Gmatrix_train_'+kernel_type+'.npy')
        test_docs = np.load('Gmatrix_test_'+kernel_type+'.npy')
    else:
        train_docs = np.load('Gmatrix_train_'+kernel_type+str(k)+'.npy')
        test_docs = np.load('Gmatrix_test_'+kernel_type+str(k)+'.npy')
    train_labels = train_labels_raw
    test_labels = test_labels_raw
else:
    train_docs, train_labels, test_docs, test_labels = gram_matrices(kernel_type,
                                            train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw, k, m_lambda)
if SAVE_MATRICES:
    if kernel_type == 'wk':
        np.save('Gmatrix_train_'+kernel_type,train_docs)
        np.save('Gmatrix_test_'+kernel_type,test_docs)
    else:
        np.save('Gmatrix_train_'+kernel_type+str(k),train_docs)
        np.save('Gmatrix_test_'+kernel_type+str(k),test_docs)

#kernel_train = kernel_wk.compute(train_docs,train_docs)
# Train the model
model = train_classifier(train_docs, train_labels) #instead of train_docs -> kernel_train

#kernel_test = kernel_wk.compute(train_docs,test_docs)
# Predict
predictions = model.predict(test_docs)  #instead of test_docs -> kernel_test
#predictions = model.predict(kernel_test)
# Evaluate predictions
evaluate(test_labels, predictions)
#
# evaluate_acq(test_labels, predictions)
# evaluate_earn(test_labels, predictions)
# evaluate_corn(test_labels, predictions)
# evaluate_crude(test_labels, predictions)
