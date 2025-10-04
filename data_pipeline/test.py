import pandas as pd

df = pd.read_csv("datasets/UAVIDS-2025.csv")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True) * 100)