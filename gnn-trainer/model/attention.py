
import torch
from torch import nn


class HanSemanticAttention(nn.Module):
    def __init__(self, in_dim, dim_a):
        super(HanSemanticAttention, self).__init__()
        self.in_dim = in_dim
        self.dim_a = dim_a

        self.project = nn.Sequential(
            nn.Linear(self.in_dim, self.dim_a),
            nn.Tanh(),
            nn.Linear(self.dim_a, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(dim=0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(dim=1)


class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        # Shape of h: (num_relations, batch_size, dim)

        attention = torch.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).squeeze()
        attention = self.dropout(attention)

        h = torch.einsum('rb,rbd->bd', attention, h)

        if return_attn:
            return h, attention
        else:
            return h
