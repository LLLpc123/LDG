import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
class GraphConv(nn.Module):

    def __init__(self,cfg, relu=True):
        super().__init__()

        if cfg.drop_rate:
            self.dropout = nn.Dropout(p=0.1)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(cfg.hidden_dim+45, cfg.hidden_dim+45))
        self.b = nn.Parameter(torch.zeros(cfg.hidden_dim+45))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        adj = adj/torch.sum(adj,dim = -1).unsqueeze(-1)
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.matmul(adj, torch.matmul(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


#gcn without normalization2
class GCN(nn.Module):

    def __init__(self,cfg, relu=True):
        super().__init__()

        if cfg.drop_rate:
            self.dropout = nn.Dropout(p=0.1)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(cfg.hidden_dim+45, cfg.hidden_dim+45))
        self.b = nn.Parameter(torch.zeros(cfg.hidden_dim+45))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        # adj = adj/torch.sum(adj,dim = -1).unsqueeze(-1)
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.matmul(adj, torch.matmul(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs
