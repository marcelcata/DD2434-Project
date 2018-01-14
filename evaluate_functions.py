from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support

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
    return precision, recall, f1

def evaluate_earn(test_labels, predictions):
    labels = [3]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("earn")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    return precision, recall, f1

def evaluate_corn(test_labels, predictions):
    labels = [1]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("corn")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    return precision, recall, f1

def evaluate_crude(test_labels, predictions):
    labels = [2]
    precision = precision_score(test_labels, predictions,labels,average='macro')
    recall = recall_score(test_labels, predictions,labels, average='macro')
    f1 = f1_score(test_labels, predictions,labels,average='macro')
    print("crude")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    return precision, recall, f1

def evaluate(test_labels, predictions):
    precision, recall, f1score, support = precision_recall_fscore_support(test_labels, predictions)
    print("eval")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(f1score))

    return precision, recall, f1score