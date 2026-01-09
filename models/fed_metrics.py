from pathlib import Path

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np

from flwr.app import MetricRecord

class FedMetrics:
    @staticmethod
    def macro_f1(labels: Tensor, predictions: Tensor) -> float:
        return float(f1_score(labels, predictions, average='macro'))

    @staticmethod
    def auroc(labels: Tensor, probabilities: Tensor) -> float:
        return float(roc_auc_score(labels, probabilities))

    @staticmethod
    def auprc(labels: Tensor, probabilities: Tensor) -> float:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        return float(auc(recall, precision))

    @staticmethod
    def recall_at_fpr(labels: Tensor, probabilities: Tensor, target_fpr: float) -> float:
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
            auroc = [metrics['auroc'] for metrics in rounds]
            plt.plot(x, auroc, label=f'Day {day}')
        plt.title(
            "Area Under Receiver Operating Characteristic Curve\n"
            "by Days per Round"
        )
        plt.xlabel('Rounds')
        plt.ylabel('AUROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_directory / 'daily_auroc.png', dpi=200)
