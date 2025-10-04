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

def partition_to_client(n_clients: int) -> None:
    df = pd.read_csv("datasets/UAVIDS-2025.csv")
    df['client_id'] = df.apply(deterministic_hash, axis=1, args=(n_clients,))
    client_dict = df.groupby("client_id").groups
    client_map = {}
    for client, index in client_dict.items():
        client_map[client] = df.loc[index, 'FlowID'].tolist()
    with open("data_pipeline/splits/clients.yaml", 'w') as file:
        yaml.dump(client_map, file, default_flow_style=False)


'''
Create Day-1...Day-4 by re-weighting class priors 
(e.g., Day-1 almost all Normal; Day-2 introduce Flooding; Day-3 add Sybil; Day-4 add Wormhole and re-balance), 
simulating concept drift without generating new traffic.

Choose which flows are active per day.
'''

# To maintain proportion with highest sample size
def min_multiplier(traffic_data: dict, labels: dict) -> int:
    multipliers = []
    for traffic_type, fraction in traffic_data.items():
        if not fraction:
            continue
        multiplier = len(labels[traffic_type]) // (fraction * 100)
        multipliers.append(multiplier)
    return min(multipliers)

def create_days() -> None:
    RNG = 0
    df = pd.read_csv("datasets/UAVIDS-2025.csv", index_col=1)
    with open("data_pipeline/splits/uavids_priors.yaml") as file:
        days: dict[str, dict[str, float]]
        days = yaml.safe_load(file)

    labels = {label: subframe for label, subframe in df.groupby('label')}
    day_flows = {}
    for day, traffic_data in days.items():
        multiplier = min_multiplier(traffic_data, labels)
        flows: list[pd.Series] = []
        for label, fraction in traffic_data.items():
            n_samples = int(fraction * 100 * multiplier)
            samples = (labels[label]
                       .sample(n_samples, random_state=RNG)['FlowID'])
            flows.append(samples)
        flow_ids = (pd.concat(flows)
                    .sample(frac=1, random_state=RNG)
                    .tolist())
        day_flows[day] = flow_ids
    with open("data_pipeline/splits/uavids_days.yaml", 'w') as file:
        yaml.dump(day_flows, file, default_flow_style=False)

if __name__ == "__main__":
    # partition_to_client(20)
    create_days()