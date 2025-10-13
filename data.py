import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from lightning import LightningDataModule

class ALSFRSDataset(Dataset):
    def __init__(self, csv_path, feature_cols=None, target_col='target'):
        self.data = pd.read_csv(csv_path)
        self.target_col = target_col
        self.feature_cols = feature_cols or [c for c in self.data.columns if c != target_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return x, y

class DataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, test_size=0.2, val_size=0.1, target_col='target'):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.target_col = target_col

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=42)

        self.train_ds = ALSFRSDataset(train_df, target_col=self.target_col)
        self.val_ds = ALSFRSDataset(val_df, target_col=self.target_col)
        self.test_ds = ALSFRSDataset(test_df, target_col=self.target_col)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

if __name__ == '__main__':
    dm = DataModule('data.csv', batch_size=16)
    dm.setup()
    for batch in dm.train_dataloader():
        x, y = batch
        print(x.shape, y.shape)
        break