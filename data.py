import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from lightning import LightningDataModule

class ALSFRSDataset(Dataset):
    def __init__(self, dataframe, feature_cols=None, target_col='Target'):
        self.dataframe = dataframe
        self.feature_cols = feature_cols if feature_cols else [col for col in dataframe.columns if col != target_col]
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        return features, target

class DataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, feature_cols=None, target_col='Target', test_size=0.2, val_size=0.1, random_state=42):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def prepare_data(self):
        self.dataframe = pd.read_csv(self.csv_path)

    def setup(self, stage=None):
        train_val_df, test_df = train_test_split(self.dataframe, test_size=self.test_size, random_state=self.random_state)
        train_df, val_df = train_test_split(train_val_df, test_size=self.val_size, random_state=self.random_state)

        self.train_dataset = ALSFRSDataset(train_df, feature_cols=self.feature_cols, target_col=self.target_col)
        self.val_dataset = ALSFRSDataset(val_df, feature_cols=self.feature_cols, target_col=self.target_col)
        self.test_dataset = ALSFRSDataset(test_df, feature_cols=self.feature_cols, target_col=self.target_col)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)