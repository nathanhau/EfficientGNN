import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from utilities.utils import getNormAugAdj,mulAdj

class DGC(nn.Module):
    precompute_dict={}
    def __init__(self,in_feats,n_classes,K,T,device,bias=True,norm=None,is_linear=False):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.T=T
        self.K=K
        # print(f"{self.K} {K}")
        self.delta=T/K
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.norm=norm
        self.nm=None
        self.device=device
        if self.norm=="ln":
            self.nm=nn.LayerNorm(n_classes)
        elif self.norm=="bn":
            self.nm=nn.BatchNorm1d(n_classes)
        # self.reset_parameters()
        self.is_linear=is_linear
        # print(DGC.precompute_dict.keys())
        # print(self.K)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self,g,feat):
        if self.precompute is None:
            self.precompute=DGC.precompute_dict.get(self.K,None)
            # print('precomp assigned')
        if self.precompute is None:
            t1=perf_counter()
            adj=g.adj().to(self.device)
            t3=perf_counter()
            S=getNormAugAdj(adj,self.device).to(self.device)
            print(f'norm: {perf_counter()-t3}')
            I=torch.eye(adj.shape[0]).to_sparse().to(self.device)
            t4=perf_counter()
            self.precompute=(1-self.delta)*I + self.delta*S
            print(f'pre: {self.precompute.is_sparse}')
            print(f'alp: {perf_counter()-t4}')
            t2=perf_counter()
            
            self.precompute=mulAdj(self.precompute,self.K)
            if (self.is_linear):
                self.precompute=torch.sparse.mm(self.precompute,feat)
            print(f'mul: {perf_counter()-t2}')
            print(f'preprocess time: {perf_counter()-t1}')
            self.precompute.to(self.device)
            DGC.precompute_dict[self.K]=self.precompute

        feat.to(self.device)
        # print(feat.device)
        # print(self.precompute.device)
        # print(feat)
        if (not self.is_linear):
            h=torch.sparse.mm(self.precompute,feat)
        else:
            h=self.precompute.detach().clone()
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h


#Input adj. matrix without self-loop
def dgc_precompute(features,adj,T,K,device):
    # print(features)
    if T==0 or K==0:
        return features,0
    features=features.to(device)
    adj=adj.to(device)
    delta=T/K
    t=perf_counter()
    S=getNormAugAdj(adj)
    I=torch.eye(adj.shape[0]).to_sparse()
    S_delta=(1-delta)*I + delta*S
    for _ in range(K):
        features=torch.sparse.mm(S_delta,features)
    precompute_time=perf_counter()-t
    # print(features,precompute_time)
    return features,precompute_time

