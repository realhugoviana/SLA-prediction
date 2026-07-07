import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from lightning import LightningDataModule

from utils import get_intervals, get_months, sort_df

# Conversion du dataframe pandas en dataset PyTorch
class ALSFRSDataset(Dataset):
    def __init__(self, dataframe, feature_cols=None, target_cols='Target'):
        self.dataframe = dataframe # DataFrame source
        self.feature_cols = feature_cols # Features définis comme tout ce qui n'est pas target
        self.target_cols = target_cols # Colonne à prédire

        self.months = get_months(self.dataframe)
        # self.dataframe = sort_df(self.dataframe, self.intervals)

    # Retourne la taille du dataset
    def __len__(self):
        return len(self.dataframe)

    # Retourne une ligne du dataset
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] 
        
        feature_list = []
        
        for month in self.months:
            feature_slice = row[self.feature_cols[self.feature_cols.str.contains(rf'_M{month}$', na=False)]]
            feature_list.append(feature_slice.reset_index(drop=True))

        features_numpy = pd.concat(feature_list, axis=1).values.astype('float32') 

        features = torch.from_numpy(features_numpy).float().T # Use .float() to ensure float32 is used
        target = torch.tensor(row[self.target_cols], dtype=torch.float32) # Conversion de la target en tensor
        return features, target

# Lecture du csv, découpage du dataset en train, val, test et chargement les données en batch
class DataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, feature_cols=None, target_cols='Target', n_folds=10, random_state=42, fold_index=0):
        super().__init__()
        self.csv_path = csv_path # Fichier csv contenant les données
        self.batch_size = batch_size # Taille de batch
        self.feature_cols = feature_cols # Noms des colonnes de features si précisé, sinon toutes sauf target
        self.target_cols = target_cols # Target
        self.n_folds = n_folds # Nombre de folds pour le cross-validation
        self.random_state = random_state # Seed
        self.fold_index = fold_index # Index de la fold pour le cross-validation
        self.scaler = StandardScaler() # Normalisation

    # Lecture du csv
    def prepare_data(self):
        self.dataframe = pd.read_csv(self.csv_path)

    # Découpage du dataset en train, val, test et normalisation des features
    def setup(self, stage=None):

        if self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state) # KFold pour le cross-validation
            folds = list(kf.split(self.dataframe)) # Création des folds
            train_indices, val_indices = folds[self.fold_index] # Récupération des indices de la fold courante
            train_df = self.dataframe.iloc[train_indices] # Création du df de train
            val_df = self.dataframe.iloc[val_indices] # Création du df de validation
        else:
            train_df, val_df = train_test_split(self.dataframe, test_size=0.1, random_state=self.random_state) # Découpage de train+val en train et val

        self.feature_cols = self.dataframe.columns[~self.dataframe.columns.str.contains(r'_M0')] # toutes sauf target
        self.target_cols = self.dataframe.columns[self.dataframe.columns.str.contains(r'_M0')]

        self.train_dataset = ALSFRSDataset(train_df, feature_cols=self.feature_cols, target_cols=self.target_cols) # Création du dataset de train
        self.val_dataset = ALSFRSDataset(val_df, feature_cols=self.feature_cols, target_cols=self.target_cols) # val

    # Charge les données en batch pour le train, val et test
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
class AutoregressiveALSFRSDataset(Dataset):
    def __init__(self, dataframe, feature_cols=None, target_cols='Target'):
        self.dataframe = dataframe # DataFrame source
        self.feature_cols = feature_cols # Features définis comme tout ce qui n'est pas target
        self.target_cols = target_cols # Colonne à prédire

        self.months = get_months(self.dataframe)
        # self.dataframe = sort_df(self.dataframe, self.intervals)

    # Retourne la taille du dataset
    def __len__(self):
        return len(self.dataframe)

    # Retourne une ligne du dataset
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] 
        
        feature_list = []
        for month in self.months:
            feature_slice = row[self.feature_cols[self.feature_cols.str.contains(rf'_M{month}$', na=False)]]
            if not feature_slice.empty:
                feature_list.append(feature_slice.reset_index(drop=True))

        features_numpy = pd.concat(feature_list, axis=1).values.astype('float32') 

        features = torch.from_numpy(features_numpy).float().T # Use .float() to ensure float32 is used
        target = torch.tensor(row[self.target_cols], dtype=torch.float32) # Conversion de la target en tensor
        return features, target

# Lecture du csv, découpage du dataset en train, val, test et chargement les données en batch
class AutoregressiveDataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, feature_cols=None, target_cols='Target'):
        super().__init__()
        self.csv_path = csv_path # Fichier csv contenant les données
        self.batch_size = batch_size # Taille de batch
        self.feature_cols = feature_cols # Noms des colonnes de features si précisé, sinon toutes sauf target
        self.target_cols = target_cols # Target
    # Lecture du csv
    def prepare_data(self):
        self.dataframe = pd.read_csv(self.csv_path)

    # Découpage du dataset en train, val, test et normalisation des features
    def setup(self, stage=None):

        self.feature_cols = self.dataframe.columns[~self.dataframe.columns.str.contains(r'Target')] # Features si précisé, sinon toutes sauf target
        self.target_cols = self.dataframe.columns[self.dataframe.columns.str.contains(r'Target')]
    

        self.test_dataset = AutoregressiveALSFRSDataset(self.dataframe, feature_cols=self.feature_cols, target_cols=self.target_cols) # test

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)