import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from models.SGC import SGC, sgc_precompute
from models.SGCRes import SGCRes
from utilities.utils import mulAdj, getNormAugAdj


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,device, bias=True, norm=None, dropout=0, iso=False, adj=None):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers[0]
        self.activation = activation
        self.bias = bias
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self.iso = iso
        self.adj = adj
        self.layers = nn.ModuleList()
        self.nm = nn.ModuleList()
        ind=self.in_feats
        outd=self.n_hidden
        for i in range(self.n_layers):
            if i>0:
                ind=self.n_hidden
            if i==self.n_layers-1:
                outd=self.n_classes
                
            self.layers.append(conv.GraphConv(ind, outd, "both", bias=self.bias))
            if i < self.n_layers-1:
                if self.norm == "bn":
                    self.nm.append(nn.BatchNorm1d(n_hidden))
                elif self.norm == "ln":
                    self.append(nn.LayerNorm(n_hidden))
            # print(outd)
        if self.iso and (self.adj != None):
            stdv = self.get_std()
            for layer in self.layers:
                for i in range(0, layer.weight.shape[0]):
                    layer.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, feat):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            h = self.layers[i](g, h)
            if i < self.n_layers-1:
                if self.norm is not None:
                    h = self.nm[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def get_std(self):
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim=1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim=0).to_dense().unsqueeze(0) + 1
        # support = torch.sqrt(torch.sum(d_i * d_j))
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        # print(stdv)
        return stdv
# rewrite to fit all linear layers
