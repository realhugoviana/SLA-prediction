import torch
import torch.nn as nn
import lightning as L
import torchmetrics
import optuna

class NN(L.LightningModule):
    def __init__(self, input_dim, output_dim, n_layer=2, n_units=16, learning_rate=1e-3):
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        # )

        self.layers = nn.Sequential()
        self.layers.add_module('input_layer', nn.Linear(input_dim, n_units))
        self.layers.add_module('input_activation', nn.ReLU())
        for i in range(n_layer - 1):
            self.layers.add_module(f'hidden_layer_{i+1}', nn.Linear(n_units, n_units))
            self.layers.add_module(f'hidden_activation_{i+1}', nn.ReLU())

        
        self.out = nn.Linear(n_units, output_dim)

        self.criterion = nn.MSELoss()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
    
    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log_dict({'train_loss': loss,
                       'train_mae': self.train_mae(y_hat, y),
                       'train_rmse': self.train_rmse(y_hat, y),
                       'train_r2': self.train_r2(y_hat, y)})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log_dict({'val_loss': loss,
                       'val_mae': self.val_mae(y_hat, y),
                       'val_rmse': self.val_rmse(y_hat, y),
                       'val_r2': self.val_r2(y_hat, y)})
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log_dict({'test_loss': loss,
                       'test_mae': self.test_mae(y_hat, y),
                       'test_rmse': self.test_rmse(y_hat, y),
                       'test_r2': self.test_r2(y_hat, y)})
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer