# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import utils as hyper
from sklearn.preprocessing import MinMaxScaler


def ROC_AUC(target1, target2, groundtruth, x):
    """
    
    :param target2d: the 2D anomaly component  
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.reshape(rows * cols)
    # minmax = MinMaxScaler()
    # result = minmax.fit_transform(target.reshape(-1, 1))
    fpr1, tpr1, thresholds = metrics.roc_curve(label, target1)
    fpr2, tpr2, thresholds = metrics.roc_curve(label, target2)
    auc1 = metrics.auc(fpr1, tpr1)
    auc2 = metrics.auc(fpr2, tpr2)
    fig = plt.figure(x)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr1, tpr1, 'b.--', label='Detected result')
    ax.plot(fpr2, tpr2, 'r.--', label='AE result')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.show()
    return auc1, auc2
