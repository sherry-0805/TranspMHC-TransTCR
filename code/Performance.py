import numpy as np
from collections import Counter
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight



def performances(y_true, y_pred, y_prob, print_ = True):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp +tn ) /(tn +fp +fn +tp)
    try:
        mcc = ((tp *tn) - (fn *fp)) / np.sqrt(np.float((tp +fn ) *(tn +fp ) *(tp +fp ) *(tn +fn)))
    except:
        print('MCC Error: ', (tp +fn ) *(tn +fp ) *(tp +fp ) *(tn +fn))
        mcc = np.nan
    try:
        recall = tp / (tp +fn)
    except:
        recall = np.nan
    try:
        sensitivity = tp /(tp +fn)
    except:
        sensitivity = np.nan
    try:
        specificity = tn /(tn +fp)
    except:
        specificity = np.nan
    try:
        precision = tp / (tp +fp)
    except:
        precision = np.nan
    try:
        f1 = 2* precision * recall / (precision + recall)
    except:
        f1 = np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)


    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity,
                                                                                              specificity, accuracy,
                                                                                              mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(precision, recall, f1))

    return (roc_auc,  accuracy, f1, mcc, sensitivity, specificity, precision, recall)


def performances_to_pd(performances_list):
    metrics_name = ['roc_auc',  'accuracy', 'f1', 'mcc', 'sensitivity', 'specificity', 'precision', 'recall']

    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)

    return performances_pd


