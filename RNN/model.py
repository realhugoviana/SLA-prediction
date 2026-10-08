import torch
import torch.nn as nn
import lightning as L
import torchmetrics

# Module lighting : pratique car pas besoin de coder à la main les boucles d'entrainement
class RNNmodel(L.LightningModule):
    def __init__(self, input_dim, output_dim, architecture, n_layer=2, n_units=16, learning_rate=1e-3, activation='relu', optimizer='Adam', criterion='MSE', loss_coef=None, regularizer="Ridge", regularization_parameter=0.0, dropout=0.0, bidirectional=False):
        super().__init__()

        # Sauvegarde des paramètres
        self.save_hyperparameters()
        self.architecture = architecture # RNN, GRU, LSTM
        self.input_dim = input_dim # Nombre de features
        self.output_dim = output_dim # Nombre de cibles à prédire
        self.n_layer = n_layer # Nombre de couches cachées
        self.n_units = n_units # Nombre de neurones par couche cachée
        self.lr = learning_rate # Learning rate
        self.dropout = dropout # Taux de dropout
        self.bidirectional = bidirectional # RNN bidirectionnelle ou non
        self.activation = activation # Fonction d'activation
        self.loss_coef = loss_coef # Coefficient de la perte
        self.regularizer = regularizer
        self.regularization_parameter = regularization_parameter

        # Architecture
        if self.architecture == "RNN":
            self.rnn = nn.RNN(input_size=input_dim, 
                            hidden_size=n_units, 
                            num_layers=n_layer, 
                            nonlinearity=activation, 
                            dropout=dropout if n_layer > 1 else 0.0, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        elif self.architecture == "GRU":
            self.rnn = nn.GRU(input_size=input_dim,
                              hidden_size=n_units,
                              num_layers=n_layer,
                              dropout=dropout if n_layer > 1 else 0.0,
                              batch_first=True,
                              bidirectional=bidirectional)
        elif self.architecture == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=n_units,
                               num_layers=n_layer,
                               dropout=dropout if n_layer > 1 else 0.0,
                               batch_first=True,
                               bidirectional=bidirectional)
        
        if not self.bidirectional:
            self.out = nn.Linear(n_units, output_dim)
        else:
            self.out = nn.Linear(2 * n_units, output_dim)

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
    def forward(self, x, hx=None):
        rnn_out, out_hx = self.rnn(x, hx) # Sortie du RNN
        last_time_step = rnn_out[:, -1, :] # On prend la sortie du dernier time step
        out = self.out(last_time_step) # Passage dans la couche de sortie
        return out, out_hx

    def regularization(self):
        if self.regularizer == "Lasso":
            return self.regularization_parameter * sum(p.abs().mean() for p in self.parameters())

        if self.regularizer == "Ridge":
            return self.regularization_parameter * sum(p.pow(2).mean() for p in self.parameters())

        if self.regularizer == "ElasticNet":
            return self.regularization_parameter * (sum(p.abs().mean() for p in self.parameters()) + sum(p.pow(2).mean() for p in self.parameters()))

        return 0
    
    # Fonction d'entrainement pour une batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_score = y[:, 0]  # On prend la première colonne comme score
        y_features = y[:, 1:]  # On prend les autres colonnes comme features
        
        y_hat, _ = self.forward(x) # Prédiction

        score_hat = y_hat[:, 0] # On prend la première sortie comme score
        features_hat = y_hat[:, 1:] # On prend les autres sorties comme features

        if y_features.shape[1] == 0:
            loss = self.criterion(score_hat, y_score) # Perte uniquement sur le score
        elif self.loss_coef is None:
            loss = self.criterion(y_hat, y) # Perte sur le score et les features
        else:
            loss = self.loss_coef * self.criterion(score_hat, y_score) + (1 - self.loss_coef) * self.criterion(features_hat, y_features) # Perte

        loss += self.regularization()

        y_hat = y_hat.view(-1, self.output_dim)
        y = y.view(-1, self.output_dim)

        self.log_dict({'train_loss': loss,
                    'train_mae': self.train_mae(y_hat, y),
                    'train_rmse': self.train_rmse(y_hat, y),
                    'train_r2': self.train_r2(y_hat, y)})
        
        return loss # Retourne la perte
    
    # Validation, pas de retro propagation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_score = y[:, 0]  # On prend la première colonne comme score
        y_features = y[:, 1:]  # On prend les autres colonnes comme features
        
        y_hat, _ = self.forward(x)

        score_hat = y_hat[:, 0] # On prend la première sortie comme score
        features_hat = y_hat[:, 1:] # On prend les autres sorties comme features

        if y_features.shape[1] == 0:
            loss = self.criterion(score_hat, y_score)
        elif self.loss_coef is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.loss_coef * self.criterion(score_hat, y_score) + (1 - self.loss_coef) * self.criterion(features_hat, y_features)

        loss += self.regularization()

        y_hat = y_hat.view(-1, self.output_dim)  
        y = y.view(-1, self.output_dim)

        self.log_dict({'val_loss': loss,
                    'val_mae': self.val_mae(y_hat, y),
                    'val_rmse': self.val_rmse(y_hat, y),
                    'val_r2': self.val_r2(y_hat, y)})
        
        return loss
    
    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_score = y[:, 0]  # On prend la première colonne comme score
        y_features = y[:, 1:]  # On prend les autres colonnes comme features
        
        y_hat, _ = self.forward(x)

        score_hat = y_hat[:, 0] # On prend la première sortie comme score
        features_hat = y_hat[:, 1:] # On prend les autres sorties comme features

        if y_features.shape[1] == 0:
            loss = self.criterion(score_hat, y_score)
        elif self.loss_coef is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.loss_coef * self.criterion(score_hat, y_score) + (1 - self.loss_coef) * self.criterion(features_hat, y_features)

        loss += self.regularization()

        y_hat = y_hat.view(-1, self.output_dim)
        y = y.view(-1, self.output_dim)

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

class AutoregressiveRNN(L.LightningModule):
    def __init__(self, model, X_scalers, ordered_features):
        super().__init__()

        self.model = model
        self.X_scalers = X_scalers
        self.features = ordered_features["feat"]

        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.test_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
    
    def forward(self, x):
        pred, _ = self.model(x)

        return pred
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        objective_metrics = []
        for i in range(y.shape[1]):
            y_hat = self.forward(x)

            for j in range(len(self.features)):
                y_hat_unscaled = torch.from_numpy(self.X_scalers[self.features[j]].inverse_transform(y_hat[:, j:j+1].detach().cpu().numpy())).to(self.device)

                self.log_dict({f'test_mae_{self.features[j]}_{i+1}': self.test_mae(y_hat_unscaled, y[:,i, j:j+1]),
                               f'test_rmse_{self.features[j]}_{i+1}': self.test_rmse(y_hat_unscaled, y[:,i, j:j+1]),
                               **({f'test_r2_{self.features[j]}_{i+1}': self.test_r2(y_hat_unscaled, y[:,i, j:j+1])}if x.size(0) >= 2 else {})})

                if j == 0:
                    objective_metrics.append(self.test_mae(y_hat_unscaled, y[:,i, j:j+1]))

            y_hat = torch.unsqueeze(y_hat, dim=1)

            x = torch.cat([x[:, 1:, :], y_hat], dim=1)

        self.log_dict({'objective_value': torch.mean(torch.stack(objective_metrics))} if objective_metrics else {})

        return objective_metrics