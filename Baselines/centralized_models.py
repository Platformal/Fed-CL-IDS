from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import yaml

def load_days() -> dict:
    with open("data_pipeline/splits/uavids_days.yaml") as file:
        return yaml.safe_load(file)

def data_from_flows(flows: list[int], dataset: pd.DataFrame) -> tuple:
    # Return row data and label
    flows_df = dataset.loc[flows]
    features = flows_df.drop('label', axis=1)
    labels = flows_df['label']
    return features.to_numpy(), labels.to_numpy()

def mlp(features, labels):
    pass

def main():
    dataset = pd.read_csv("datasets/UAVIDS-2025 Preprocessed.csv").set_index('FlowID')
    model = MLPClassifier(
        random_state=0,
    )
    for day, flow_ids in load_days().items():
        features, labels = data_from_flows(flow_ids, dataset)
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, random_state=0, test_size=0.2)
        model.fit(train_features, train_labels)
        model.predict(test_features)
        print(model.score(test_features, test_labels))
        # break

if __name__ == "__main__":
    main()