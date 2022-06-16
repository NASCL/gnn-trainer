
from torch import nn
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling

from model.provgem import PROVGEM
from model.han import HAN
from model.rgcn import RGCN


PROVGRAPH_RELATIONS = [
    'file-wasDerivedFrom-file',
    'file-wasDerivedFrom-iattr',
    'file-wasDerivedFrom-link',
    'file-wasDerivedFrom-path',
    'file-wasDerivedFrom-pipe',
    'file-wasDerivedFrom-process_memory',
    'file-wasGeneratedBy-task',
    'iattr-wasGeneratedBy-task',
    'link-wasDerivedFrom-link',
    'link-wasDerivedFrom-path',
    'link-wasGeneratedBy-task',
    'machine-wasAssociatedWith-task',
    'packet-wasDerivedFrom-socket',
    'pipe-wasDerivedFrom-pipe',
    'pipe-wasGeneratedBy-task',
    'process_memory-wasDerivedFrom-argv',
    'process_memory-wasDerivedFrom-file',
    'process_memory-wasDerivedFrom-path',
    'process_memory-wasDerivedFrom-process_memory',
    'process_memory-wasDerivedFrom-socket',
    'process_memory-wasGeneratedBy-task',
    'shm-wasGeneratedBy-task',
    'socket-wasDerivedFrom-address',
    'socket-wasDerivedFrom-packet',
    'socket-wasDerivedFrom-process_memory',
    'socket-wasDerivedFrom-socket',
    'socket-wasDerivedFrom-task',
    'socket-wasGeneratedBy-task',
    'task-used-file',
    'task-used-link',
    'task-used-pipe',
    'task-used-process_memory',
    'task-used-socket',
    'task-wasInformedBy-task',
]


class ProvGraphClassifier(nn.Module):
    def __init__(
            self,
            conv,
            feat_dim,
            embed_dim,
            dim_a,
            aggregate='mean',
            dropout=0.,
            num_layers=2,
            activation=None,
            norm=False,
            layer_norm=False,
            bias=True,
            self_loop=True,
            num_heads=1,
            num_classes=1,
    ):
        super(ProvGraphClassifier, self).__init__()
        self.relations = PROVGRAPH_RELATIONS
        self.num_relations = len(self.relations)
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.aggregate = aggregate.casefold()
        self.dropout = dropout
        self.activation = activation.casefold()
        self.norm = norm

        self.bias = bias
        self.self_loop = self_loop
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.embedder = nn.ModuleList()
        for i in range(self.num_layers):
            self.embedder.append(
                self._get_conv_module(
                    conv,
                    in_dim=self.feat_dim if i == 0 else self.embed_dim,
                    out_dim=self.embed_dim
                )
            )

        self.readout_fn = self._get_reduce_fn(aggregate)

        self.classifier = MLPClassifier(self.embed_dim, self.num_classes)

    def forward(self, graph, feat):
        h = feat
        for layer in self.embedder:
            h = layer(graph, h)

        h = self.readout_fn(graph, h)
        return self.classifier(h)

    @staticmethod
    def _get_reduce_fn(agg_type):
        if agg_type == 'mean':
            readout_fn = AvgPooling()
        elif agg_type == 'max':
            readout_fn = MaxPooling()
        elif agg_type == 'sum':
            readout_fn = SumPooling()
        else:
            raise ValueError('Invalid aggregation function')

        return readout_fn

    def _get_conv_module(self, conv, in_dim, out_dim):
        conv = conv.casefold()
        if conv == 'provgem':
            return PROVGEM(
                relations=self.relations,
                feat_dim=in_dim,
                embed_dim=out_dim,
                dim_a=self.dim_a,
                aggregate=self.aggregate,
                dropout=self.dropout,
                activation=self.activation,
                norm=self.norm,
            )
        elif conv == 'han':
            return HAN(
                relations=self.relations,
                feat_dim=in_dim,
                embed_dim=out_dim,
                dim_a=self.dim_a,
                aggregate=self.aggregate,
                dropout=self.dropout,
                activation=self.activation,
                norm=self.norm,
                num_heads=self.num_heads,
            )
        elif conv == 'rgcn':
            return RGCN(
                relations=self.relations,
                feat_dim=in_dim,
                embed_dim=out_dim,
                bias=self.bias,
                activation=self.activation,
                self_loop=self.self_loop,
                dropout=self.dropout,
                layer_norm=self.layer_norm
            )
        else:
            raise ValueError('Invalid graph conv module.')


class MLPClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=1):
        super(MLPClassifier, self).__init__()
        self.embed_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
