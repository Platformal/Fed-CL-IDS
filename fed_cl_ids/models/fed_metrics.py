from pathlib import Path

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np

from flwr.app import MetricRecord

class FedMetrics:
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
        interpolated_recall = np.interp(target_fpr, fpr, tpr)
        return float(interpolated_recall)

    @staticmethod
    def create_metric_plots(
        daily_metrics: list[list[MetricRecord]],
        save_directory: Path
    ) -> None:
        # List for each day
        # For each day there are x rounds that contain a dict[str, float]
        n_rounds = len(daily_metrics[0])
        x = list(range(1, n_rounds + 1))
        for day, rounds in enumerate(daily_metrics, 1):
            roc_auc = [metrics['roc-auc'] for metrics in rounds]
            plt.plot(x, roc_auc, label=f'Day {day}')
        plt.title(
            "Area Under Receiver Operating Characteristic Curve\n" \
            "of Days per Round"
        )
        plt.xlabel('Rounds')
        plt.ylabel('AUROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_directory / 'daily_roc_auc.png')
