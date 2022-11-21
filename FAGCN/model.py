import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv


class FAGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 dropout, eps, device, layer_num=2):
        super().__init__()
        
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FAConv(hidden_channels, self.eps, dropout))
        
        self.linear1 = nn.Linear(in_channels, hidden_channels, device=device)
        self.linear2 = nn.Linear(hidden_channels, out_channels, device=device)
        self.reset_parameters()
        
    def reset_parameters(self):
        # TODO: 这是干嘛的？
        nn.init.xavier_normal_(self.linear1.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear2.weight, gain=1.414)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        # MLP
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        raw = x
        for i in range(self.layer_num):
            x = self.layers[i](x, raw, edge_index)
            x = self.eps * raw + x
        x = self.linear2(x)
        return F.log_softmax(x, dim=1) # dim=1是对行做归一化
        