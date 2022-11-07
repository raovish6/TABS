import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, hidden_dim, embedding_dim)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
