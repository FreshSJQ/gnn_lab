import torch
import pickle
import numpy as np
import scipy.sparse as sp

import os.path as osp
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, LINKXDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Load homophily and heterophily graph datasets
def load_data(dataset_name):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        print(data)
    elif dataset_name in ['chameleon', 'squirrel']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WikipediaNetwork')
        dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WebKB')
        dataset = WebKB(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
    elif dataset_name in ['Actor']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Actor')
        dataset = Actor(path, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        print(data)
    elif dataset_name in ['penn94']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'LINKXDataset')
        dataset = LINKXDataset(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        print(data)
    else:
        print(f"not find dataset {dataset_name}")
        exit(0)
    
    adj_orig = to_dense_adj(data.edge_index)[0]
    gcn_norm = T.GCNNorm(True)
    data = gcn_norm(data)
    adj_norm = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_weight)[0]

    labels = data.y
    nclass = dataset.num_classes
    
    if data.train_mask.ndim == 1:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        # 暂时只选择第1个mask，后续可以考虑其他的
        train_mask = data.train_mask[:, 0]
        val_mask = data.val_mask[:, 0]
        test_mask = data.test_mask[:, 0]

    return data.x.to(device), adj_orig.to(device), adj_norm.to(device), labels.to(device), train_mask, val_mask, test_mask, nclass