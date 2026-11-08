import pandas as pd


def load_train(path="data/train.csv"):
    return pd.read_csv(path)