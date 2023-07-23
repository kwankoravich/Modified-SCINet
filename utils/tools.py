from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm
import pandas as pd

def train_test_split(df:pd.DataFrame, ratio:int):
    train_df = df.iloc[: round(df.shape[0] * ratio)]
    val_df = df.drop(train_df.index)
    return train_df, val_df

def normalize(method, train_df):
    if method == "minmax":
        scaler = MinMaxScaler()
    if method == "standard":
        scaler = StandardScaler()

    train_df_norm = train_df.copy()
    cols = train_df.columns
    scaler.fit(train_df[cols])
    train_df_norm[cols] = scaler.transform(train_df[cols])

    return train_df_norm, scaler
