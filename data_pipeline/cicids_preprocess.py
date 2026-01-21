from pathlib import Path
from typing import cast
import random

import pandas as pd
import numpy as np

THURSDAY_AFTERNOON_PATH = Path("datasets/CIC-IDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
DIRECTORY = Path("datasets/CIC-IDS2017")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# To parquet & encode labels
def preprocess() -> None:
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
    # df = pd.concat(map(pd.read_csv, DIRECTORY.iterdir()))
    # df = pd.read_parquet(DIRECTORY / 'temp.parquet')
    df = (
        pd.read_csv(THURSDAY_AFTERNOON_PATH)
          .replace([-np.inf, np.inf], np.nan)
          .dropna()
          .drop_duplicates()
    )
    df.columns = df.columns.str.casefold().str.strip()
    df['label'] = df['label'].map(generalized_mapping).map(label_encoding)
    df = df.astype('float32')

    # print(df.describe())
    # Index included?
    # df.to_csv("data_pipeline/preprocessed_cicids/1.csv")
    df.to_parquet("data_pipeline/preprocessed_cicids/1.parquet")

if __name__ == '__main__':
    # preprocess()
    print(pd.read_parquet("data_pipeline/preprocessed_cicids/1.parquet"))