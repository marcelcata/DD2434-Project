from nltk.corpus import reuters
import random
import pickle

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

filename = 'dataset/subset_470.pkl'

def generate_dataset(shuffle=True):
    if shuffle:
        print('Shuffling dataset')
    dataset = {}
    for split in SPLITS:
        dataset[split] = []
        for category in CATEGORIES:
            print('Loading "{}" split "{}" category ({} examples)'.format(split, category, SIZES[split][category]))
            # File names are preceded by 'train' or 'test'
            raw_docs = [reuters.raw(doc_id) for doc_id in reuters.fileids(category) if split in doc_id]
            if shuffle:
                random.shuffle(raw_docs)
            counter = 0
            for raw_doc in raw_docs:
                if len(raw_doc) > 15:
                    dataset[split].append((raw_doc, category))
                    counter += 1
                if counter >= SIZES[split][category]:
                    break
    return

dataset = generate_dataset()
pickle.dump(dataset, open(filename, 'wb'))

