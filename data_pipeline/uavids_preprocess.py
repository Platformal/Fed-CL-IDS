from sklearn.preprocessing import RobustScaler
import pandas as pd
import yaml

RANDOM_SEED = 0

def generate_days() -> None:
    df = pd.read_csv("fed_cl_ids/datasets/UAVIDS-2025.csv", index_col=1)
    with open("fed_cl_ids/data_pipeline/splits/uavids_distribution.yaml") as file:
        days: dict[str, dict[str, float]] = yaml.safe_load(file)

    labels = {label: subframe for label, subframe in df.groupby('label')}
    day_flows = {}
    for day, flow_ids in enumerate(days.values(), 1):
        multiplier = min_multiplier(flow_ids, labels)
        flows: list[pd.Series] = []
        for label, fraction in flow_ids.items():
            n_samples = int(fraction * 100 * multiplier)
            samples = labels[label].sample(
                n_samples,
                random_state=RANDOM_SEED
            )
            flows.append(samples['FlowID'])
        flow_data = (
            pd.concat(flows)
            .sample(frac=1, random_state=RANDOM_SEED)
            .tolist()
        )
        day_flows[day] = flow_data
    with open("fed_cl_ids/data_pipeline/splits/uavids_days.yaml", 'w') as file:
        yaml.dump(day_flows, file, default_flow_style=False)

# To maintain proportion with the highest sample size
def min_multiplier(traffic_data: dict, labels: dict) -> int:
    multipliers = []
    for traffic_type, fraction in traffic_data.items():
        if not fraction:
            continue
        multiplier: int = len(labels[traffic_type]) // (fraction * 100)
        multipliers.append(multiplier)
    return min(multipliers)

def preprocess_uavids():
    labels = {
        'Normal Traffic': 0, 
        'Sybil Attack': 1, 
        'Flooding Attack': 2,
        'Wormhole Attack': 3, 
        'Blackhole Attack': 4
    }
    initial_df = pd.read_csv("fed_cl_ids/datasets/UAVIDS-2025.csv")
    dropped = ['SrcAddr', 'DstAddr', 'Protocol']
    main_df = initial_df.drop(dropped, axis=1).set_index('FlowID')
    for column in main_df.columns[:-1]:
        main_df[column] = RobustScaler().fit_transform(
            main_df[column].to_numpy().reshape((-1, 1))
        )
    main_df['label'] = main_df['label'].map(labels)
    main_df.to_csv("fed_cl_ids/datasets/UAVIDS-2025 Preprocessed.csv")

if __name__ == "__main__":
    # generate_clients(20)
    # generate_days()
    preprocess_uavids()