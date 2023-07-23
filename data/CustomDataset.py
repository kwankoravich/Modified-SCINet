from torch.utils.data import Dataset
import pandas as pd
import torch

class CustomDatasets(Dataset):
    # def __init__(self, df, dfy, seq_len, pred_len):
    def __init__(self, df, seq_len, pred_len):
        self.df = df.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dfx = df.copy()
        self.dfx.iloc[:,-1] = self.dfx.iloc[:,-1].shift(2)
        self.dfx = self.dfx.dropna()
        # self.dfx.close = self.dfx.close
        # display(self.dfx[self.dfx.iloc[:,-1] != self.dfx.iloc[:,-1]])
        # display(self.dfx)
        # self.dfy = dfy


    def __getitem__(self, idx):
        # input
        # idx+=1
        x_start = idx
        # x_end = idx + self.seq_len
        x_end = idx + self.seq_len # For percentage change

        y_start = x_end
        # y_end = y_start + self.pred_len
        y_end = y_start + self.pred_len # For percentage change

        x = torch.tensor(self.dfx.iloc[x_start:x_end].values)
        # x = torch.tensor(self.df.iloc[x_start:x_end].pct_change().dropna().values)
        # y = torch.tensor(self.df.iloc[y_start:y_end].pct_change().dropna().values)
        # y = torch.tensor(self.df.iloc[y_start:y_end, -1].values)
        y = torch.tensor(self.df.iloc[y_start:y_end].values)
        # print(x)
        # print(y)

        return x.float(), y.float()

    def __len__(self):
        return len(self.df) - (self.seq_len + self.pred_len)