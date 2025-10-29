from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import precision_recall_curve, auc
from numpy import ndarray, interp

class Losses:
    @staticmethod
    def macro_f1(labels: ndarray, predictions: ndarray) -> float:
        return float(f1_score(labels, predictions, average='macro'))

    @staticmethod
    def roc_auc(labels: ndarray, probabilities: ndarray) -> float:
        return float(roc_auc_score(labels, probabilities))

    @staticmethod
    def pr_auc(labels: ndarray, probabilities: ndarray) -> float:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        return float(auc(recall, precision))

    @staticmethod
    def recall_at_fpr(labels: ndarray, probabilities: ndarray, target_fpr: float) -> float:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        interpolated_recall = interp(target_fpr, fpr, tpr)
        return float(interpolated_recall)