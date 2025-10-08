from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as score

import pandas as pd
import yaml

def load_days() -> dict:
    with open("data_pipeline/splits/uavids_days.yaml") as file:
        return yaml.safe_load(file)

def flow_data(flows: list[int], dataset: pd.DataFrame) -> tuple:
    # Return row data and label
    flows_df = dataset.loc[flows]
    features = flows_df.drop('label', axis=1)
    labels = flows_df['label']
    return features.to_numpy(), labels.to_numpy()

def add_scores(day: str, scores: dict, true_labels, predicted_labels) -> None:
    pr_curve = score.precision_recall_curve(true_labels, predicted_labels)
    precision, recall, _ = pr_curve
    pr_auc = score.auc(recall, precision)
    roc_auc = score.roc_auc_score(true_labels, predicted_labels)
    f1 = score.f1_score(true_labels, predicted_labels)
    scores[day] = {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'F1': f1
    }

def main():
    dataset = (pd.read_csv("datasets/UAVIDS-2025 Preprocessed.csv")
               .set_index('FlowID'))
    model = MLPClassifier(random_state=0)
    day_scores = {}
    for day, flow_ids in load_days().items():
        features, labels = flow_data(flow_ids, dataset)
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, random_state=0, test_size=0.2)
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        add_scores(day, day_scores, test_labels, predictions)
    with open("baselines/mlp_baselines.yaml", 'w') as file:
        yaml.dump(day_scores, file, default_flow_style=False)

if __name__ == "__main__":
    main()