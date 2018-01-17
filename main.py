from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import kernel_wk
import kernel_ngk
import kernel_ssk
import evaluate_functions
import combined_kernels
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


########################### PREPROCESSING METHODS
def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def initialize(i):
    filename = 'dataset/subset_100_' + str(i) + '.pkl'
    #filename = 'dataset/subset_470_' + str(i) + '.pkl'
    filename2 = 'dataset/subset_ssk.pkl'
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

def gram_matrices(kernel_type, train_docs, train_labels, test_docs, test_labels,n, k, m_lambda):

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
                train_matrix[i][j] = kernel_ngk.compute(train_docs[i],train_docs[j], n)
            elif kernel_type == 'ssk':
                train_matrix[i][j] = kernel_ssk.compute(train_docs[i],train_docs[j], k, m_lambda)

    train_matrix = train_matrix + train_matrix.T + np.eye(n_train)

    for i in range(n_test):
        print(i)
        for j in range(n_train):
            if kernel_type == 'wk':
                test_matrix[i][j] = kernel_wk.compute(test_docs[i],train_docs[j])
            elif kernel_type == 'ngram':
                test_matrix[i][j] = kernel_ngk.compute(test_docs[i],train_docs[j], n)
            elif kernel_type == 'ssk':
                test_matrix[i][j] = kernel_ssk.compute(test_docs[i],train_docs[j], k, m_lambda)

    return train_matrix, train_labels, test_matrix, test_labels


############################ CLASSIFIER
def train_classifier(train_matrix, train_labels):
    classifier = SVC(kernel='precomputed')
    classifier.fit(train_matrix, train_labels)
    return classifier

###################### MAIN #####################


for i in range(10):
    # Prepare Dataset
    train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw = initialize(i+1)
    # Get the feature according to the chosen kernel
    kernel_type = 'ssk'
    k = 3
    n = 5
    m_lambda = 0.5

    train_docs_raw = [preprocessing(doc) for doc in train_docs_raw]
    test_docs_raw = [preprocessing(doc) for doc in test_docs_raw]

    if LOAD_MATRICES:
        if kernel_type == 'wk':
            train_matrix = np.load('gram_matrices/Subset140/Gmatrix_train_'+kernel_type+'_run'+str(i)+'.npy')
            test_matrix = np.load('gram_matrices/Subset140/Gmatrix_test_'+kernel_type+'_run'+str(i)+'.npy')
        if kernel_type == 'ngram':
            train_matrix = np.load('gram_matrices/Subset140/Gmatrix_train_' + kernel_type+str(n) +'_run'+str(i)+ '.npy')
            test_matrix = np.load('gram_matrices/Subset140/Gmatrix_test_' + kernel_type+str(n) +'_run'+str(i)+ '.npy')
        if kernel_type == 'ssk':
            train_matrix = np.load('gram_matrices/Subset140/Gmatrix_train_'+kernel_type+str(k)+'_'+str(m_lambda)+'_run'+str(i)+'.npy')
            test_matrix = np.load('gram_matrices/Subset140/Gmatrix_test_'+kernel_type+str(k)+'_'+str(m_lambda)+'_run'+str(i)+'.npy')
        train_labels = train_labels_raw
        test_labels = test_labels_raw
    else:
        train_matrix, train_labels, test_matrix, test_labels = gram_matrices(kernel_type,
                                                train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw,n, k, m_lambda)
    if SAVE_MATRICES:
        if kernel_type == 'wk':
            np.save('gram_matrices/Subset140/Gmatrix_train_'+kernel_type+'_run'+str(i),train_matrix)
            np.save('gram_matrices/Subset140/Gmatrix_test_'+kernel_type+'_run'+str(i),test_matrix)
        if kernel_type == 'ngram':
            np.save('gram_matrices/Subset140/Gmatrix_train_'+kernel_type+str(n)+'_run'+str(i),train_matrix)
            np.save('gram_matrices/Subset140/Gmatrix_test_'+kernel_type+str(n)+'_run'+str(i),test_matrix)
        if kernel_type == 'ssk':
            np.save('gram_matrices/Subset140/Gmatrix_train_' + kernel_type + str(k)+'_'+str(m_lambda)+'_run'+str(i), train_matrix)
            np.save('gram_matrices/Subset140/Gmatrix_test_' + kernel_type + str(k)+'_'+str(m_lambda)+'_run'+str(i), test_matrix)

    # Train the model
    model = train_classifier(train_matrix, train_labels)
    # Predict
    predictions = model.predict(test_matrix)

    # Evaluate predictions
    category = 'corn' #choose category to evaluate
    if category == 'acq':
        precision, recall, f1score = evaluate_functions.evaluate_acq(test_labels, predictions)
    if category == 'corn':
        precision, recall, f1score = evaluate_functions.evaluate_corn(test_labels, predictions)
    if category == 'crude':
        precision, recall, f1score = evaluate_functions.evaluate_crude(test_labels, predictions)
    if category == 'earn':
        precision, recall, f1score = evaluate_functions.evaluate_earn(test_labels, predictions)

    # Averaging:
    if i == 0:
        precision_array = np.array([precision])
        recall_array = np.array([recall])
        f1score_array = np.array([f1score])
    else:
        precision_array=np.append(precision_array,[precision])
        recall_array=np.append(recall_array,[recall])
        f1score_array=np.append(f1score_array,[f1score])
## Mean ##
mean_p = np.mean(precision_array,axis=0)
mean_r = np.mean(recall_array,axis=0)
mean_f = np.mean(f1score_array,axis=0)
print("Evaluation of 10 runs")
print('precision mean: {}'.format(mean_p))
print('recall mean: {}'.format(mean_r))
print('fscore mean: {}'.format(mean_f))
## Standard deviation ##
std_p = np.std(precision_array,axis=0)
std_r = np.std(recall_array,axis=0)
std_f = np.std(f1score_array,axis=0)
print('precision std: {}'.format(std_p))
print('recall std: {}'.format(std_r))
print('fscore std: {}'.format(std_f))