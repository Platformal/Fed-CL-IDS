from numpy import ndarray, interp
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import precision_recall_curve, auc

def roc_auc(labels: ndarray, predictions: ndarray) -> float:
    return float(roc_auc_score(labels, predictions))

def pr_auc(labels: ndarray, predictions: ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return float(auc(recall, precision))

def macro_f1(labels: ndarray, predictions: ndarray) -> float:
    return float(f1_score(labels, predictions, average='macro'))

def recall_at_fpr(labels: ndarray, probabilities: ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    i = np.argmax(fpr >= target_fpr)
    recall = tpr[i]
    return float(recall)