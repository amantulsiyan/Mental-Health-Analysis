import pandas as pd

df = pd.read_csv("../data/train.csv")

print(df["label"].value_counts())