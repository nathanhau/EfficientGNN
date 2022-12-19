import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from utilities.utils import mulAdj, getNormAugAdj

class SGC(nn.Module):
    precompute_dict={}
    def __init__(self,in_feats,n_classes,K,device,bias=True,norm=None,is_linear=False):
        super(SGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.norm=norm
        self.nm=None
        self.device=device
        self.is_linear=is_linear
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
            self.precompute=SGC.precompute_dict.get(self.K,None)
            # print(self.precompute)
        if self.precompute is None:
            t1=perf_counter()
            adj=getNormAugAdj(g.adj()).to(self.device)
            self.precompute=mulAdj(adj, self.K).to(self.device)
            
            if (self.is_linear):
                self.precompute=torch.sparse.mm(self.precompute,feat)
            # print(f'preprocess time: {perf_counter()-t1}')
            SGC.precompute_dict[self.K]=self.precompute
            # print(self.precompute.size())
            
        t1=perf_counter()
        if (not self.is_linear):
            h=torch.sparse.mm(self.precompute,feat)
            # print(f'mul: {perf_counter()-t1}')
            h.to(self.device)
        else:
            h=self.precompute.detach().clone()
        # self.precompute.to(self.device)
        # print(h.size())
        t1=perf_counter()
        h=self.fc(h)
        # print(f'lin: {perf_counter()-t1}')
        if self.norm is not None:
            # print('NORM: ', self.norm)
            h=self.nm(h)
        return h


def sgc_precompute(features,adj,K,device):
    t=perf_counter()
    features=features.to(device)
    adj=adj.to(device)
    for i in range(K):
        features=torch.spmm(adj,features)
    precompute_time=perf_counter()-t
    return features,precompute_time
