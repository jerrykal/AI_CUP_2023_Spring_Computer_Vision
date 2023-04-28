"""
See section 4.3 of the MonoTrack paper:
https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf
"""

import torch.nn as nn


class HitNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        # Take the last token embedding of GRU output
        x = x[:, -1, :]

        x = self.dropout(x)
        x = self.output(x)

        return x
