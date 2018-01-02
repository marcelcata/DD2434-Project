from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

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

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

def preprocessing(document):
  doc = document.lower()   # Make all lowercase
  #Re.sub: Return the string obtained by replacing the leftmost non-overlapping occurrences of [^a-z] in string by the replacement ' '.
  doc = re.sub('[^a-z]', ' ', doc)  # Keep only words containing letters, and replace the others with space
  doc = re.sub(' +', ' ', doc)   # Get rid of multiple adjacent spaces
  # doc = re.sub('[^a-z]+')
  # Split on whitespace
  word_list = doc.split(' ')
  # Get rid of all english stopwords
  word_list = [word for word in word_list if word not in stopwords.words('english')]
  # Join with space
  preprocessed_doc = ' '.join(word_list)
 # print('Preprocessed: ', preprocessed_doc)
  # Strip of whitespace
  return preprocessed_doc.strip()

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
       # preprocessed_doc = preprocessing(raw_doc)
        # Get rid of documents that only contain the date!
        #if len(preprocessed_doc) > 15:
        if len(raw_doc) > 15:
          dataset[split].append((raw_doc, category))
          counter += 1
      for raw_id in raw_ids:
        if len(raw_id) > 15:
          dataset_ids[split].append((raw_id, category))
        if counter >= SIZES[split][category]:
          break
  return dataset,dataset_ids

def represent(documents, dataset,dataset_ids):
    # Training Set(9,603 docs): LEWISSPLIT = "TRAIN"; TOPICS = "YES"
    # Test Set(3,299 docs): LEWISSPLIT = "TEST"; TOPICS = "YES"
    # Unused(8,676 docs):   LEWISSPLIT = "NOT-USED"; TOPICS = "YES"  or TOPICS = "NO" or TOPICS = "BYPASS"


    # train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    # test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    #
    # train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    # test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    train_docs_id = map(lambda tuple: tuple[0], dataset_ids[TRAIN])
    test_docs_id = map(lambda tuple: tuple[0], dataset_ids[TEST])
    train_docs = map(lambda tuple: tuple[0], dataset[TRAIN])
    test_docs =  map(lambda tuple: tuple[0], dataset[TEST])


    # Tokenisation
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    # Transform multilabel labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])


    return (vectorised_train_documents, train_labels, vectorised_test_documents, test_labels)


def train_classifier(train_docs, train_labels):
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(train_docs, train_labels)
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


# documents = reuters.fileids()
# train_docs, train_labels, test_docs, test_labels = represent(documents)
# model = train_classifier(train_docs, train_labels)
# predictions = model.predict(test_docs)
# evaluate(test_labels, predictions)

documents = reuters.fileids()
dataset,dataset_ids = load_dataset()
train_docs, train_labels, test_docs, test_labels = represent(documents,dataset,dataset_ids)
model = train_classifier(train_docs, train_labels)
predictions = model.predict(test_docs)
evaluate(test_labels, predictions)