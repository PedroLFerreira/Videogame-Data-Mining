"""tools for ipython notebook"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import itertools
import numpy as np
from sklearn.model_selection import cross_validate

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=cm.Blues,
                          figsize=(15,12)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.tight_layout()


def plot_correlation_matrix(cm, attributes, title='Correlation Matrix', cmap='RdBu', figsize=(12,12)):
    plt.figure(figsize=figsize)
    #plt.matshow(eucorr.as_matrix(), cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(-1,1)
    tick_marks = np.arange(len(attributes))
    plt.xticks(tick_marks, attributes, rotation=45)
    plt.yticks(tick_marks, attributes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()

def gen_log_space(limit, n):
    """ generate integers in logaritmic space"""
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=int)

def cross_validate_model(model, x_train, y_train):
    m = cross_validate(model, x_train, y_train, return_train_score=True)
    
    test_score, train_score = m['test_score'], m['train_score']
    
    return (np.mean(test_score), np.std(test_score)),(np.mean(train_score), np.std(train_score))