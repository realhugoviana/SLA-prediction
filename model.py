import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

class NN(L.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            F.ReLU(),
            nn.Linear(64, 64),
            F.ReLU()
        )
        self.out = nn.Linear(64, output_dim)

        self.criterion = nn.MSELoss()
        
    
    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._common_step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._common_step(batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._common_step(batch, batch_idx)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('loss', loss)
        return loss, y, y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer