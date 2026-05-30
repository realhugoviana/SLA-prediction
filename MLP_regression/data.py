import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from lightning import LightningDataModule

# Conversion du dataframe pandas en dataset PyTorch
class ALSFRSDataset(Dataset):
    def __init__(self, dataframe, feature_cols=None, target_col='Target'):
        self.dataframe = dataframe # DataFrame source
        self.feature_cols = feature_cols if feature_cols else [col for col in dataframe.columns if col != target_col] # Features définis comme tout ce qui n'est pas target
        self.target_col = target_col # Colonne à prédire

    # Retourne la taille du dataset
    def __len__(self):
        return len(self.dataframe)

    # Retourne une ligne du dataset
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] # Récupère la ligne par son indice
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32) # Conversion des features en tensors
        target = torch.tensor(row[self.target_col], dtype=torch.float32) # Conversion de la target en tensor
        return features, target

# Lecture du csv, découpage du dataset en train, val, test et chargement les données en batch
class DataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, feature_cols=None, target_col='Target', test_size=0.2, val_size=0.1, random_state=42):
        super().__init__()
        self.csv_path = csv_path # Fichier csv contenant les données
        self.batch_size = batch_size # Taille de batch
        self.feature_cols = feature_cols # Noms des colonnes de features si précisé, sinon toutes sauf target
        self.target_col = target_col # Target
        self.test_size = test_size # Fraction du dataset pour le test
        self.val_size = val_size # Fraction du dataset pour la validation
        self.random_state = random_state # Seed
        self.scaler = StandardScaler() # Normalisation

    # Lecture du csv
    def prepare_data(self):
        self.dataframe = pd.read_csv(self.csv_path)

    # Découpage du dataset en train, val, test et normalisation des features
    def setup(self, stage=None):
        train_val_df, test_df = train_test_split(self.dataframe, test_size=self.test_size, random_state=self.random_state) # Découpage du df en train+Val et test
        train_df, val_df = train_test_split(train_val_df, test_size=self.val_size, random_state=self.random_state) # Découpage de train+val en train et val

        # self.feature_cols = self.feature_cols if self.feature_cols else [col for col in self.dataframe.columns if col != self.target_col] # Features si précisé, sinon toutes sauf target
        # train_df[self.feature_cols] = self.scaler.fit_transform(train_df[self.feature_cols]) # Normalisation des features sur le train
        # val_df[self.feature_cols] = self.scaler.transform(val_df[self.feature_cols]) # Normalisation de val en utilisant le scaler de train
        # test_df[self.feature_cols] = self.scaler.transform(test_df[self.feature_cols]) # Même chose pour le test

        self.train_dataset = ALSFRSDataset(train_df, feature_cols=self.feature_cols, target_col=self.target_col) # Création du dataset de train
        self.val_dataset = ALSFRSDataset(val_df, feature_cols=self.feature_cols, target_col=self.target_col) # val
        self.test_dataset = ALSFRSDataset(test_df, feature_cols=self.feature_cols, target_col=self.target_col) # test

    # Charge les données en batch pour le train, val et test
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)