from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support
import numpy as np

cachedStopWords = stopwords.words("english")

ACQ = 'acq'
CORN = 'corn'
CRUDE = 'crude'
EARN = 'earn'
TRAIN = 'train'
TEST = 'test'
CATEGORIES = [ACQ, CORN, CRUDE, EARN]
SPLITS = [TRAIN, TEST]

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
def initialize(shuffle = True):
    dataset = load_dataset(shuffle)
    # train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    # test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    #
    # train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    # test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    labels = {
        'acq': 0,
        'corn': 1,
        'crude': 2,
        'earn': 3
    }
    # train_docs_id = map(lambda tuple: tuple[1], dataset[TRAIN])
    # test_docs_id = map(lambda tuple: tuple[1], dataset[TEST])
    # train_docs = map(lambda tuple: tuple[0], dataset[TRAIN])
    # test_docs =  map(lambda tuple: tuple[0], dataset[TEST])

    #select documents and labels
    train_docs = [tuple[0] for tuple in dataset[TRAIN]]
    train_labels = np.array([labels[tuple[1]] for tuple in dataset[TRAIN]])
    test_docs = [tuple[0] for tuple in dataset[TEST]]
    test_labels = np.array([labels[tuple[1]] for tuple in dataset[TEST]])
    return train_docs, train_labels, test_docs, test_labels

def load_dataset(shuffle=True):
  if shuffle:
    print('Shuffling dataset')
  dataset = {}
  dataset_ids = {}
  for split in SPLITS:
    dataset[split] = []
    dataset_ids[split] = []
    for category in CATEGORIES:
      print('Loading "{}" split "{}" category ({} examples)'.format(split, category, SIZES[split][category]))
      # File names are preceded by 'train' or 'test'
      raw_docs = [reuters.raw(doc_id) for doc_id in reuters.fileids(category) if split in doc_id]
      raw_ids = [reuters.fileids() for doc_id in reuters.fileids(category) if split in doc_id]
      if shuffle:
        random.shuffle(raw_docs)
      counter = 0
      for raw_doc in raw_docs:
        if len(raw_doc) > 15:
          dataset[split].append((raw_doc, category))
          counter += 1
      for raw_id in raw_ids:
        if len(raw_id) > 15:
          dataset_ids[split].append((raw_id, category))
        if counter >= SIZES[split][category]:
          break
  return dataset


########################### USING THE KERNELS
def extract_features(kernel, train_docs, train_labels, test_docs, test_labels):

    # Tokenisation
    if kernel == 'wk':
        vectorizer = TfidfVectorizer(tokenizer=word_kernel)
    elif kernel == 'ngram':
        vectorizer = TfidfVectorizer(tokenizer=ngram_kernel)
    elif kernel == 'ssk':
        vectorizer = TfidfVectorizer(tokenizer=ssk_kernel)
    else:
        print('Kernel not found.')

    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    # Transform multilabel labels
    # mlb = MultiLabelBinarizer()
    # train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    # test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])
    return (vectorised_train_documents, train_labels, vectorised_test_documents, test_labels)

# WORD KERNEL
def word_kernel(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

# N-GRAM KERNEL
def ngram_kernel():
    # TO DO
    return None

# SSK KERNEL
def ssk_kernel():
    # TO DO
    return None



############################ CLASSIFIER
def train_classifier(train_docs, train_labels):
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(train_docs, train_labels)
    return classifier


############################ EVALUATION FUNCTIONS
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
kernel = 'wk'
train_docs, train_labels, test_docs, test_labels = extract_features(kernel,
                                            train_docs_raw, train_labels_raw, test_docs_raw, test_labels_raw)
# Train the model
model = train_classifier(train_docs, train_labels)

# Predict
predictions = model.predict(test_docs)

# Evaluate predictions
evaluate(test_labels, predictions)


#evaluate_acq(test_labels, predictions)
#evaluate_earn(test_labels, predictions)
#evaluate_corn(test_labels, predictions)
#evaluate_crude(test_labels, predictions)
#evaluate(test_labels, predictions)