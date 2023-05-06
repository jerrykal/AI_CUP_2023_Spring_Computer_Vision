import torch
import torch.nn as nn


class ShotTypeClassifier(nn.Module):
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
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        sequence_lengths = torch.sum(x[:, :, 0] != -1, dim=1).detach().cpu().numpy()

        x_embedded = self.embedding(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_embedded, sequence_lengths, batch_first=True, enforce_sorted=False
        )

        out_packed, _ = self.rnn(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        out = out[torch.arange(out.size(0)), sequence_lengths - 1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out
