import torch
import torch.nn as nn

class lstm(nn.Module):  
    def __init__(
        self,
        seq_input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
    ):
        super(lstm, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions

        self.projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  
        else:
            h_last = h_n[-1]  

        return self.projection(h_last)  