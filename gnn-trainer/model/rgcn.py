
import torch
from torch import nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import AvgPooling
from dgl.utils import expand_as_pair


class RGCN(nn.Module):
    def __init__(self,
                 relations,
                 feat_dim,
                 embed_dim,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super(RGCN, self).__init__()
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.bias = bias
        self.activation = self._get_activation_fn(activation)
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # add weights
        self.weight = nn.Parameter(torch.Tensor(self.num_relations, self.feat_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(embed_dim))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(embed_dim, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(feat_dim, embed_dim))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        self.readout_fn = AvgPooling()

    def forward(self, graph, feat):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.embed_dim, device=graph.device)
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self.self_loop:
                loop_message = torch.matmul(feat_dst, self.loop_weight)

            for i, rel in enumerate(self.relations):
                if rel in graph.etypes:
                    graph.srcdata['h'] = feat_src

                    graph.update_all(
                        fn.copy_u('h', 'm'),
                        fn.sum('m', 'neigh'),
                        etype=rel
                    )

                    h[i] = torch.matmul(graph.dstdata['neigh'], self.weight[i])

            h = torch.mean(h, dim=0)

            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + loop_message
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)

        return h
    
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
