
import torch
from torch import nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling
from dgl.utils import expand_as_pair

from model.attention import SemanticAttention


class PROVGEM(nn.Module):
    def __init__(
            self,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            aggregate='mean',
            dropout=0.,
            activation=None,
            norm=False
    ):
        super(PROVGEM, self).__init__()
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=True)

        self.transform = nn.ModuleDict({
            rel: MessageTransform(
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dropout=self.dropout,
                activation=self.activation,
                norm=self.norm
            )
            for rel in relations
        })

        self.attention = SemanticAttention(self.num_relations, self.embed_dim, self.dim_a)

        self.reduce_fn, self.readout_fn = self._get_reduce_fn(aggregate)

    @staticmethod
    def _get_reduce_fn(agg_type):
        if agg_type == 'mean':
            reduce_fn = fn.mean
            readout_fn = AvgPooling()
        elif agg_type == 'max':
            reduce_fn = fn.max
            readout_fn = MaxPooling()
        elif agg_type == 'sum':
            reduce_fn = fn.sum
            readout_fn = SumPooling()
        else:
            raise ValueError('Invalid aggregation function')

        return reduce_fn, readout_fn

    def forward(self, graph, feat):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.embed_dim, device=graph.device)
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            for i, rel in enumerate(self.relations):
                if rel in graph.etypes:
                    graph.srcdata['h'] = feat_src

                    graph.update_all(
                        fn.copy_u('h', 'm'),
                        self.reduce_fn('m', 'neigh'),
                        etype=rel
                    )

                    h_rel = feat_dst + graph.dstdata['neigh']

                    h[i] = self.transform[rel](h_rel)

            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.attention(h)

        return h


class MessageTransform(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_layers=2,
            dropout=0.,
            activation='relu',
            norm=False,
    ):
        super(MessageTransform, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True) if norm else None

        self.layers = nn.ModuleList([
            nn.Linear(self.in_dim, self.out_dim) if i == 0
            else nn.Linear(self.out_dim, self.out_dim)
            for i in range(num_layers)
        ])

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = F.relu
        elif activation == 'elu':
            act_fn = F.elu
        elif activation == 'gelu':
            act_fn = F.gelu
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)

            if self.norm is not None:
                x = self.norm(x)
            if self.activation is not None:
                x = self.activation(x)

        return x
