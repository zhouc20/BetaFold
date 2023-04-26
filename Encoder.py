import torch
import torch.nn as nn

class Amino_Acid(nn.Module):

    def __init__(self, hid_dim):
        super(Amino_Acid, self).__init__()

        self.encoder = nn.Embedding(28, hid_dim)

    def forward(self, x):
        return self.encoder(x)
