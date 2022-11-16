import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

class SGC(nn.Module):
    def __init__(self,in_feats,n_classes,K,bias=True,norm=None):
        super(SGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.norm=norm
        self.nm=None
        if self.norm=="ln":
            self.nm=nn.LayerNorm(n_classes)
        elif self.norm=="bn":
            self.nm=nn.BatchNorm1d(n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self,g,feat):
        if self.precompute is None:
            adj=g.adj()
            self.precompute=mulAdj(adj, self.K)
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        print(self.norm)
        if self.norm is not None:
            h=self.nm(h)
        return h


def sgc_precompute(features,adj,K):
    t=perf_counter()
    for i in range(K):
        features=torch.spmm(adj,features)
    precompute_time=perf_counter()-t
    return features,precompute_time
