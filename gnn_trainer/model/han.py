#!/usr/bin/env python3

import dgl
import torch
from torch import nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
import dgl.function as fn
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling

from model.attention import HanSemanticAttention


class HAN(nn.Module):
    def __init__(
            self,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            num_heads=1,
            aggregate='sum',
            dropout=0.,
            activation=None,
            norm=False
    ):
        super(HAN, self).__init__()
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.num_heads = num_heads

        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=True)

        self.convs = nn.ModuleDict({
            rel: GATConv(
                in_feats=self.feat_dim,
                out_feats=self.embed_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                allow_zero_in_degree=True
            )
            for rel in relations
        })
        self.attention = HanSemanticAttention(
            in_dim=self.embed_dim * self.num_heads, dim_a=dim_a
        )

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
        z = torch.zeros(graph.num_nodes(), self.num_relations, self.embed_dim, device=graph.device)
        with graph.local_scope():
            for i, rel in enumerate(self.relations):
                if rel in graph.etypes:
                    rel_graph = graph[dgl.NTYPE, rel, dgl.NTYPE]
                    z[:, i] = self.convs[rel](rel_graph, feat).squeeze()

            z = self.attention(z)

        return z
