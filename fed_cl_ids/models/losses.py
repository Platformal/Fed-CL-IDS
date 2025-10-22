from numpy import ndarray, searchsorted
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import precision_recall_curve, recall_score, auc
# ROC‑AUC, PR‑AUC, macro‑F1, Recall@FPR=1%

def roc_auc(labels: ndarray, predictions: ndarray) -> float:
    return float(roc_auc_score(labels, predictions))

def pr_auc(labels: ndarray, predictions: ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return float(auc(recall, precision))

def macro_f1(labels: ndarray, predictions: ndarray) -> float:
    return float(f1_score(labels, predictions, average='macro'))

def recall_per_1fpr(labels: ndarray, probabilities: ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    i = searchsorted(fpr, 0.01, side='right') - 1
    if not (-1 < i < len(tpr)):
        raise IndexError
    return float(tpr[i])