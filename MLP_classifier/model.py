import torch
import torch.nn as nn
import lightning as L
import torchmetrics

class MLP_classifier(L.LightningModule):
    def __init__(self, input_dim, n_layer=2, n_units=16, learning_rate=1e-3, decroissant=False, activation='ReLU', optimizer='Adam'):
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate

        self.activation_fn = {
            'ReLU': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }[activation]

        self.layers = nn.Sequential()
        self.layers.add_module('input_layer', nn.Linear(input_dim, n_units))
        self.layers.add_module('input_activation', self.activation_fn)

        if decroissant:
            in_units = n_units
            out_units = in_units // 2
            for i in range(n_layer - 1):
                self.layers.add_module(f'hidden_layer_{i+1}', nn.Linear(in_units, out_units))
                self.layers.add_module(f'hidden_activation_{i+1}', self.activation_fn)
                in_units = out_units
                out_units = in_units // 2
            self.out = nn.Linear(out_units*2, 49)
        else:
            for i in range(n_layer - 1):
                self.layers.add_module(f'hidden_layer_{i+1}', nn.Linear(n_units, n_units))
                self.layers.add_module(f'hidden_activation_{i+1}', self.activation_fn)
            self.out = nn.Linear(n_units, 49)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = {
            'Adam': torch.optim.Adam,
            'RMSprop': torch.optim.RMSprop,
            'Adagrad': torch.optim.Adagrad
        }[optimizer]

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
        y = y.view(-1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if len(y) > 1:
            self.log_dict({'train_loss': loss,
                        'train_mae': self.train_mae(y_hat.argmax(dim=1), y),
                        'train_rmse': self.train_rmse(y_hat.argmax(dim=1), y),
                        'train_r2': self.train_r2(y_hat.argmax(dim=1), y)})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if len(y) > 1:
            self.log_dict({'val_loss': loss,
                        'val_mae': self.val_mae(y_hat.argmax(dim=1), y),
                        'val_rmse': self.val_rmse(y_hat.argmax(dim=1), y),
                        'val_r2': self.val_r2(y_hat.argmax(dim=1), y)})
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if len(y) > 1:
            self.log_dict({'test_loss': loss,
                        'test_mae': self.test_mae(y_hat.argmax(dim=1), y),
                        'test_rmse': self.test_rmse(y_hat.argmax(dim=1), y),
                        'test_r2': self.test_r2(y_hat.argmax(dim=1), y)})
        return loss
    
    def configure_optimizers(self):
        optimizer =  self.optimizer(self.parameters(), lr=self.lr)
        return optimizer