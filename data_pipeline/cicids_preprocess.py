from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

PREPROCESSED_DIR = Path("data_pipeline/preprocessed_cicids")
DATASET_PATH = Path("datasets/CIC-IDS2017")
RANDOM_SEED = 0

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# To parquet & encode labels
def preprocess() -> None:
    important_columns = [
        ' Average Packet Size',
        'Flow Bytes/s',
        ' Max Packet Length',
        ' Packet Length Mean',
        ' Fwd Packet Length Mean',
        ' Subflow Fwd Bytes',
        ' Fwd IAT Min',
        ' Avg Fwd Segment Size',
        'Total Length of Fwd Packets',
        ' Flow IAT Mean',
        ' Fwd Packet Length Max',
        ' Fwd IAT Std',
        ' Fwd Header Length',
        ' Flow Duration',
        ' Flow Packets/s',
        ' Fwd IAT Max',
        'Fwd Packets/s',
        ' Flow IAT Std',
        'Fwd IAT Total',
        ' Fwd IAT Mean',
        ' Label'
    ]
    generalized_mapping = {
        'BENIGN': 'Normal',

        'DoS Hulk': 'DoS',
        'DDoS': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',

        'Web Attack � Brute Force': 'Web Attack',
        'Web Attack � XSS': 'Web Attack',
        'Web Attack � Sql Injection': 'Web Attack',

        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',

        'PortScan': 'Probe',
        'Bot': 'Botnet',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed',
    }
    label_encoding = {
        'Normal': 0,
        'DoS': 1,
        'Web Attack': 2,
        'Brute Force': 3,
        'Probe': 4,
        'Botnet': 5,
        'Infiltration': 6,
        'Heartbleed': 7
    }

    # If need to create main df
    # df = pd.concat(map(pd.read_csv, DATASET_PATH.glob('*.csv')))
    intermediate = Path("data_pipeline/temp_cicids.parquet")
    # df.to_parquet(intermediate)

    df = (
        pd.read_parquet(intermediate, columns=important_columns)
          .replace([-np.inf, np.inf], np.nan)
          .dropna()
          .drop_duplicates()
    )

    # Downcasting ints and floats to 32-bits
    int_col = df.select_dtypes(include='integer').columns
    df[int_col] = df[int_col].apply(pd.to_numeric, downcast='integer')

    float_col = df.select_dtypes(include='float').columns
    df[float_col] = df[float_col].astype('float32')

    # Encode labels
    df.columns = df.columns.str.casefold().str.strip()
    df['label'] = (
        df['label'].map(generalized_mapping)
        .map(label_encoding)
        .astype('uint8')
    )

    # 70/30 Ratio 0:1
    n_positives = (df['label'] != 0).sum()
    total_samples = int(n_positives / 0.3)
    n_negatives = total_samples - n_positives
    labels_df = df[df['label'] == 0].sample(n_negatives, random_state=RANDOM_SEED)
    df = pd.concat((df[df['label'] != 0], labels_df))

    # Even stratification
    groups = tuple(StratifiedKFold(4).split([''] * len(df), df['label']))
    for day, (_, day_indices) in enumerate(groups, 1):
        output_file = PREPROCESSED_DIR / f'{day}.parquet'
        day_df: pd.DataFrame = df.iloc[day_indices]
        day_df.to_parquet(output_file)
        print(day_df['label'].value_counts())

if __name__ == '__main__':
    preprocess()