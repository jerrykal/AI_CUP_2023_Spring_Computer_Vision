import torch
import torch.nn as nn

class BackhandClassifier(nn.Module):
    def __init__(
        self,
        input_size=11,
        hidden_size=128,
        num_layers=2,
        num_classes=9,
        dropout_prob=0.5,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_layers),
            num_layers=2,
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        sequence_lengths = torch.sum(x[:, :, 0] != -1, dim=1).detach().cpu().numpy()

        x_embedded = self.embedding(x)

        # Transformer expects input as (S,N,E), reshaping the input
        x_transformer = x_embedded.permute(1, 0, 2)

        out = self.transformer_encoder(x_transformer)

        # Getting output for the last token in the sequence
        out = out[sequence_lengths - 1, torch.arange(out.size(1)), :]
        
        out = self.dropout(out)
        out = self.fc(out)

        return out
