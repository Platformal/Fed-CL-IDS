from pathlib import Path
from typing import cast
import random

from sklearn.preprocessing import RobustScaler
import pandas as pd
import yaml

RANDOM_SEED = 0
UAVIDS_CSV_PATH = Path("datasets/UAVIDS-2025.csv")
DISTRIBUTION_PATH = Path("data_pipeline/splits/uavids_distribution.yaml")
UAVIDS_DAYS_PATH = Path("data_pipeline/splits/uavids_days.yaml")
OUTPUT_DIR_PATH = Path("data_pipeline/preprocessed_uavids")

def generate_uavids_days() -> None:
    """
    Assigns list of integers for each day in file. It is split by
    anchoring Normal Traffic and sampling other attacks from that
    """
    df = pd.read_csv(UAVIDS_CSV_PATH)
    # Group flow IDs by their labels
    groups: dict[str, list[int]] = {
        str(label): list(map(lambda flow_id: flow_id - 1, grouped_df['FlowID']))
        for label, grouped_df in df.groupby("label")
    }

    # Print distribution percentage
    total_flows = sum(len(value) for value in groups.values())
    print(f"Total Number of Flows: {total_flows:,}")
    for key, value in groups.items():
        print(f"{key}: {len(value)} flows, {len(value) / total_flows:.3%}")

    # Initially split the Normal Traffic across days,
    # then add values to add up to percentage labeled in uavids_distribution
    # Normal Traffic 100% will be utilized, other attacks may not

    with DISTRIBUTION_PATH.open(encoding='utf-8') as file:
        distribution = cast(dict[int, dict[str, float]], yaml.safe_load(file))

    days = {
        1: 0.4,
        2: 0.25,
        3: 0.22,
        4: 0.13
    }

    # Final output should be a day with traffic types holding lists of ints
    uavids_days: dict[int, dict[str, list[int]]] = {
        day: {}
        for day in range(1, 5)
    }

    print(f"Normal Traffic Sum: {sum(days.values()):.2%}")
    n_normal = len(groups['Normal Traffic'])
    for day, percent in days.items():
        print(f"Day {day}:")
        # Calculate total flows per day in respect to normal day's traffic percent
        n_day_normal = int(n_normal * percent)
        ideal_total_flows = n_day_normal / distribution[day]['Normal Traffic']
        for traffic_type in groups:
            # Calculate how many flows to get in respect to total size
            n_traffic_type = ideal_total_flows * distribution[day][traffic_type]
            indices = random.sample(groups[traffic_type], int(n_traffic_type))
            groups[traffic_type] = list(set(groups[traffic_type]).difference(indices))
            uavids_days[day][traffic_type] = uavids_days[day].get(traffic_type, [])
            uavids_days[day][traffic_type].extend(indices)
        n_daily_flows = sum(
            len(uavids_days[day][traffic])
            for traffic in uavids_days[day]
        )
        print(f"\tDaily Total Flows: {n_daily_flows:,}", )
        for traffic, flows in uavids_days[day].items():
            print(f"\t{traffic}: {len(flows):,}")

    # Reduce uavids_days into one list[int] for each day
    final_dict: dict[int, list[int]] = {}
    for day, traffic_types in uavids_days.items():
        final_dict[day] = [
            flow
            for flows in traffic_types.values()
            for flow in flows
        ]

    print("\nLeftover Traffic Flows:")
    for traffic, flows in groups.items():
        print(f"\t{traffic}: {len(flows):,}")
    n_final_flows = sum(len(item) for item in final_dict.values())
    print(f"Percent Used: {n_final_flows / total_flows:.2%}")

    with UAVIDS_DAYS_PATH.open('w', encoding='utf-8') as file:
        yaml.dump(final_dict, file, default_flow_style=False)

def preprocess_uavids():
    with UAVIDS_DAYS_PATH.open(encoding='utf-8') as file:
        uavids_days = cast(dict[int, list[int]], yaml.safe_load(file))

    labels = {
        'Normal Traffic': 0,
        'Sybil Attack': 1,
        'Flooding Attack': 2,
        'Wormhole Attack': 3,
        'Blackhole Attack': 4
    }
    initial_df = pd.read_csv(UAVIDS_CSV_PATH)
    dropped = ['SrcAddr', 'DstAddr', 'Protocol']
    main_df = initial_df.drop(dropped, axis=1).set_index('FlowID')
    main_labels = main_df.pop('label')
    for column in main_df.columns:
        main_df[column] = RobustScaler().fit_transform(
            main_df[column]
            .to_numpy()
            .reshape((-1, 1))
        )
        main_df['label'] = main_labels.map(labels)
    print(main_df)

    for day, flows in uavids_days.items():
        day_df = main_df.loc[flows]
        day_df.to_parquet(OUTPUT_DIR_PATH / f"{day}.parquet")
        print(day_df['label'].value_counts() / len(day_df))
if __name__ == "__main__":
    generate_uavids_days()
    preprocess_uavids()