import torch
import torch.nn as nn
import lightning as L
import torchmetrics

# Module lighting : pratique car pas besoin de coder à la main les boucles d'entrainement
class MLP_regressor(L.LightningModule):
    def __init__(self, input_dim, output_dim, n_layer=2, n_units=16, learning_rate=1e-3, decroissant=False, activation='ReLU', optimizer='Adam', criterion='MSE'):
        super().__init__()

        # Sauvegarde des paramètres
        self.save_hyperparameters()

        self.lr = learning_rate # Learning rate

        # Sélection de la fonction d'activation en fonction de l'entrée
        self.activation_fn = {
            'ReLU': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }[activation]

        # Architecture
        self.layers = nn.Sequential()
        self.layers.add_module('input_layer', nn.Linear(input_dim, n_units)) # Première couche 
        self.layers.add_module('input_activation', self.activation_fn) # Couche d'activation

        # Si décroissant, on divise par deux le nombre de neurone par couche à chaque couche
        if decroissant:
            in_units = n_units
            out_units = in_units // 2
            for i in range(n_layer - 1): # On créer autant de couche que précisé en entrée
                self.layers.add_module(f'hidden_layer_{i+1}', nn.Linear(in_units, out_units))
                self.layers.add_module(f'hidden_activation_{i+1}', self.activation_fn)
                in_units = out_units
                out_units = in_units // 2
            self.out = nn.Linear(out_units*2, output_dim) # Dernière couche sans activation 
        else:
            for i in range(n_layer - 1):
                self.layers.add_module(f'hidden_layer_{i+1}', nn.Linear(n_units, n_units))
                self.layers.add_module(f'hidden_activation_{i+1}', self.activation_fn)
            self.out = nn.Linear(n_units, output_dim)

        # Fonction de perte
        self.criterion = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'Huber': nn.SmoothL1Loss()
        }[criterion]

        # Optimizer (déscente de gradient & retro-propagation)
        self.optimizer = {
            'Adam': torch.optim.Adam,
            'RMSprop': torch.optim.RMSprop,
            'Adagrad': torch.optim.Adagrad
        }[optimizer]

        # Métriques : MAE, RMSE, R2
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
    # Fonction de passe dans le NN
    def forward(self, x):
        x = self.layers(x) # Passe par toutes les couches
        x = self.out(x) # Puis la couche de sortie
        return x
    
    # Fonction d'entrainement pour une batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x) # Prédiction
        loss = self.criterion(y_hat, y) # Perte
        if len(y) > 1: # Log
            self.log_dict({'train_loss': loss,
                        'train_mae': self.train_mae(y_hat, y),
                        'train_rmse': self.train_rmse(y_hat, y),
                        'train_r2': self.train_r2(y_hat, y)})
        return loss # Retourne la perte
    
    # Validation, pas de retro propagation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if len(y) > 1:
            self.log_dict({'val_loss': loss,
                        'val_mae': self.val_mae(y_hat, y),
                        'val_rmse': self.val_rmse(y_hat, y),
                        'val_r2': self.val_r2(y_hat, y)})
        return loss
    
    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if len(y) > 1:
            self.log_dict({'test_loss': loss,
                        'test_mae': self.test_mae(y_hat, y),
                        'test_rmse': self.test_rmse(y_hat, y),
                        'test_r2': self.test_r2(y_hat, y)})
        return loss
    
    # Configuration de l'optimizer
    def configure_optimizers(self):
        optimizer =  self.optimizer(self.parameters(), lr=self.lr) # Prend en entrée les paramètres et le learning rate
        return optimizer
    
    def on_train_epoch_end(self):
        if self.device.type == 'mps':
            torch.mps.empty_cache()