import torch
import torch.nn as nn
import lightning as L
import torchmetrics

# Module lighting : pratique car pas besoin de coder à la main les boucles d'entrainement
class RNNmodel(L.LightningModule):
    def __init__(self, input_dim, output_dim, architecture, n_layer=2, n_units=16, learning_rate=1e-3, activation='relu', optimizer='Adam', criterion='MSE', weight_decay=0.0, dropout=0.0, bidirectional=False):
        super().__init__()

        # Sauvegarde des paramètres
        self.save_hyperparameters()
        self.architecture = architecture # RNN, GRU, LSTM
        self.input_dim = input_dim # Nombre de features
        self.output_dim = output_dim # Nombre de cibles à prédire
        self.n_layer = n_layer # Nombre de couches cachées
        self.n_units = n_units # Nombre de neurones par couche cachée
        self.lr = learning_rate # Learning rate
        self.weight_decay = weight_decay # Poids de régularisation L2
        self.dropout = dropout # Taux de dropout
        self.bidirectional = bidirectional # RNN bidirectionnelle ou non
        self.activation = activation # Fonction d'activation

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
    
    # Fonction d'entrainement pour une batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1, 1)
        y_hat, _ = self.forward(x) # Prédiction
        loss = self.criterion(y_hat, y) # Perte
        y_hat = y_hat.view(-1, self.output_dim)  # 17 = ton nombre de sorties
        y = y.view(-1, self.output_dim)
        self.log_dict({'train_loss': loss,
                    'train_mae': self.train_mae(y_hat, y),
                    'train_rmse': self.train_rmse(y_hat, y),
                    'train_r2': self.train_r2(y_hat, y)})
        return loss # Retourne la perte
    
    # Validation, pas de retro propagation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1, 1)
        y_hat, _ = self.forward(x)
        loss = self.criterion(y_hat, y)
        y_hat = y_hat.view(-1, self.output_dim)  # 17 = ton nombre de sorties
        y = y.view(-1, self.output_dim)
        # if len(y) > 1:
        #     # Assurez-vous que les dimensions sont compatibles
        #     if y_hat.dim() == 2 and y.dim() == 2:
        #         self.log_dict({'val_loss': loss,
        #                     'val_mae': self.val_mae(y_hat, y),
        #                     'val_rmse': self.val_rmse(y_hat, y),
        #                     'val_r2': self.val_r2(y_hat, y)})
        #     else:
        #         # Si les dimensions ne correspondent pas, reshapez
        #         if y_hat.dim() == 1 and y.dim() == 1:
        #             y_hat = y_hat.unsqueeze(0)
        #             y = y.unsqueeze(0)
        #         elif y_hat.dim() == 2 and y.dim() == 1:
        #             y = y.unsqueeze(1)
        self.log_dict({'val_loss': loss,
                    'val_mae': self.val_mae(y_hat, y),
                    'val_rmse': self.val_rmse(y_hat, y),
                    'val_r2': self.val_r2(y_hat, y)})
        return loss
    
    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1, 1)
        y_hat, _ = self.forward(x)
        loss = self.criterion(y_hat, y)
        y_hat = y_hat.view(-1, self.output_dim)  # 17 = ton nombre de sorties
        y = y.view(-1, self.output_dim)
        self.log_dict({'test_loss': loss,
                    'test_mae': self.test_mae(y_hat, y),
                    'test_rmse': self.test_rmse(y_hat, y),
                    'test_r2': self.test_r2(y_hat, y)})
        return loss

    # Configuration de l'optimizer
    def configure_optimizers(self):
        optimizer =  self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) # Prend en entrée les paramètres et le learning rate
        return optimizer
    
    def on_train_epoch_end(self):
        if self.device.type == 'mps':
            torch.mps.empty_cache()

class AutoregressiveRNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.r2 = torchmetrics.R2Score()
    
    def forward(self, x, hx=None):
        out, out_hx = self.model(x, hx)
        
        score_out = out[:, 0]

        return out, out_hx, score_out
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat, hx, score_hat = self.forward(x)

        self.log_dict({'test_loss_1': self.mae(score_hat, y[:,0]),
                    'test_mae_1': self.mae(score_hat, y[:,0]),
                    'test_rmse_1': self.rmse(score_hat, y[:,0]),
                    **({'test_r2_1': self.r2(score_hat, y[:, 0])} if x.size(0) >= 2 else {})})
        
        if y.shape[1] > 1:
            for i in range(1, y.shape[1]):
                y_hat, hx, score_hat = self.forward(y_hat.unsqueeze(1), hx)
                self.log_dict({f'test_loss_{i+1}': self.mae(score_hat, y[:,i]),
                            f'test_mae_{i+1}': self.mae(score_hat, y[:,i]),
                            f'test_rmse_{i+1}': self.rmse(score_hat, y[:,i]),
                            **({f'test_r2_{i+1}': self.r2(score_hat, y[:, i])} if x.size(0) >= 2 else {})})

        return self.mae(score_hat, y[:,0])