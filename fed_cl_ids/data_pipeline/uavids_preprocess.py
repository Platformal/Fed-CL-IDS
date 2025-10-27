from sklearn.preprocessing import RobustScaler
import pandas as pd
import hashlib
import yaml

# Map UAVIDS-2025.csv flows to clients.yaml
# Via deterministic hashing (src, dst, protocol)
def deterministic_hash(row: pd.Series, n_clients: int) -> int:
    key = f"{row['SrcAddr']}{row['DstAddr']}{row['Protocol']}"
    hash_str = hashlib.sha256(key.encode()).hexdigest()
    client_id = int(hash_str, 16) % n_clients
    return client_id

def generate_clients(n_clients: int) -> None:
    df = pd.read_csv("fed_cl_ids/datasets/UAVIDS-2025.csv")
    df['client_id'] = df.apply(deterministic_hash, axis=1, args=(n_clients,))
    client_dict = df.groupby("client_id").groups
    client_map = {
        client: df.loc[index, 'FlowID'].tolist()
        for client, index in client_dict.items()
    }
    # for client, index in client_dict.items():
    #     client_map[client] = df.loc[index, 'FlowID'].tolist()
    with open("fed_cl_ids/data_pipeline/splits/clients.yaml", 'w') as file:
        yaml.dump(client_map, file, default_flow_style=False)


'''
Create Day-1...Day-4 by re-weighting class priors 
(e.g., Day-1 almost all Normal; Day-2 introduce Flooding; Day-3 add Sybil; Day-4 add Wormhole and re-balance), 
simulating concept drift without generating new traffic.

Choose which flows are active per day.
'''

# To maintain proportion with the highest sample size
def min_multiplier(traffic_data: dict, labels: dict) -> int:
    multipliers = []
    for traffic_type, fraction in traffic_data.items():
        if not fraction:
            continue
        multiplier: int = len(labels[traffic_type]) // (fraction * 100)
        multipliers.append(multiplier)
    return min(multipliers)

def generate_days() -> None:
    RNG = 0
    df = pd.read_csv("fed_cl_ids/datasets/UAVIDS-2025.csv", index_col=1)
    with open("fed_cl_ids/data_pipeline/splits/uavids_distribution.yaml") as file:
        days: dict[str, dict[str, float]] = yaml.safe_load(file)

    labels = {label: subframe for label, subframe in df.groupby('label')}
    day_flows = {}
    for day, flow_ids in days.items():
        multiplier = min_multiplier(flow_ids, labels)
        flows: list[pd.Series] = []
        for label, fraction in flow_ids.items():
            n_samples = int(fraction * 100 * multiplier)
            samples = labels[label].sample(n_samples, 
                                           random_state=RNG)['FlowID']
            flows.append(samples)
        flow_data = (pd.concat(flows)
                    .sample(frac=1, random_state=RNG)
                    .tolist())
        day_flows[day] = flow_data
    with open("fed_cl_ids/data_pipeline/splits/uavids_days.yaml", 'w') as file:
        yaml.dump(day_flows, file, default_flow_style=False)

def preprocess():
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
    preprocess()
    pass