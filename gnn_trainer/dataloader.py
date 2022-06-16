
import os
from itertools import compress

import torch
from torch.utils.data import Dataset
import dgl


class ProvGraphDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        g = self.graphs[item]
        if self.labels is not None:
            lbl = self.labels[item]
            return g, lbl
        else:
            return g
    
    def __repr__(self) -> str:
        return f'Num graphs: {len(self.graphs)} | Labels: {torch.bincount(self.labels.int())}'

    @classmethod
    def build_dataset(cls, data_dir, attack_type, multiclass=False, test_size=None, bidirected=True):
        if attack_type == 'all':
            graphs, labels = [], []
            for i, attack in enumerate(os.listdir(data_dir)):
                graphs_path = os.path.join(data_dir, attack)
                indiv_graphs, graph_dict = dgl.load_graphs(graphs_path)

                graphs.extend(indiv_graphs)

                indiv_labels = graph_dict['labels']
                if multiclass:
                    indiv_labels[indiv_labels.nonzero(as_tuple=True)] += i
                    indiv_labels = indiv_labels.long()
                else:
                    indiv_labels = indiv_labels.float()  # BCEloss expects float
                
                labels.append(indiv_labels)  # Cross entropy loss expects long-int
            
            labels = torch.cat(labels)
            
        else:
            graphs_path = os.path.join(data_dir, f'{attack_type}-graphs-new.bin')
            graphs, graph_dict = dgl.load_graphs(graphs_path)

            labels = graph_dict['labels'].float()  # BCEloss expects float
        
        if bidirected:
            graphs = [
                dgl.to_simple(dgl.to_bidirected(graph, copy_ndata=True), copy_ndata=True, return_counts=None)
                for graph in graphs
            ]

        if test_size:
            test_mask = torch.rand(len(graphs)) < test_size
            train_mask = ~test_mask

            train_dataset = cls(list(compress(graphs, train_mask)), labels[train_mask])
            test_dataset = cls(list(compress(graphs, test_mask)), labels[test_mask])

            return train_dataset, test_dataset
        else:
            return cls(graphs, labels)
