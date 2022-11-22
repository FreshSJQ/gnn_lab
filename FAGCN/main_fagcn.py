import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, LINKXDataset, Amazon
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
from model import FAGCN

from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--layer_num', type=int, default=2)

parser.add_argument('--RPMAX', type=int, default=10)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'FAGCN-{args.dataset}', epochs=args.epochs, eps=args.eps,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)
elif args.dataset in ['chameleon', 'squirrel']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WikipediaNetwork')
    dataset = WikipediaNetwork(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WebKB')
    dataset = WebKB(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
elif args.dataset in ['Actor']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Actor')
    dataset = Actor(path, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)
    exit(0)
elif args.dataset in ['penn94']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'LINKXDataset')
    dataset = LINKXDataset(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)
elif args.dataset in ['Computers', 'Photo']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')
    dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)
    exit(0)
else:
    print(f"not find dataset {args.dataset}")
    exit(0)
 
    
def run_exp(args, dataset, data):
    def train(model, optimizer, data, train_mask):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)
    
    @torch.no_grad()
    def test(model, data, train_mask, val_mask, test_mask):
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        
        accs = []
        for mask in [train_mask, val_mask, test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs
    
    fagcn_net = FAGCN(dataset.num_features, args.hidden_channels, dataset.num_classes,
                      args.dropout, args.eps, device, args.layer_num).to(device)
    optimizer = torch.optim.Adam(fagcn_net.parameters(), lr=0.005, weight_decay=args.weight_decay)

    # 选择mask
    if data.train_mask.ndim == 1:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        # 暂时只选择第1个mask，后续可以考虑其他的
        train_mask = data.train_mask[:, 0]
        val_mask = data.val_mask[:, 0]
        test_mask = data.test_mask[:, 0]
    

    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(fagcn_net, optimizer, data, train_mask)
        train_acc, val_acc, tmp_test_acc = test(fagcn_net, data, train_mask, val_mask, test_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = tmp_test_acc
        # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    return final_test_acc


if __name__ == '__main__':
    results = []
    # for RP in tqdm(range(args.RPMAX)):
    for RP in range(args.RPMAX):
        test_acc = run_exp(args, dataset, data)
        results.append(test_acc)
        log(RP=RP, Test_Acc=test_acc)
    
    test_acc_mean = np.mean(results) * 100
    test_acc_std = np.sqrt(np.var(results)) * 100
    print(f'{args.dataset}: Test Acc Mean = {test_acc_mean:.2f} \t Test Acc Std = {test_acc_std:.2f}')