import torch
import torch.nn as nn

class StaticMLPBranch(nn.Module):
    def __init__(self, input_dim=20, embedding_dim=32, dropout_rate=0.3):
        super(StaticMLPBranch, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x)
