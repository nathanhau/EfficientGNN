import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from utilities.utils import mulAdj,getNormAugAdj
from models.DGC import DGC

class SGCRes(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,device,bias=True,norm=None,is_linear=False):
        super(SGCRes,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.alpha=alpha
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.feat_ori=None
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

    def forward(self,g,feat,feat_ori=None):
        if self.feat_ori is None:
            if feat_ori is None:
                self.feat_ori=feat
            else:
                self.feat_ori=feat_ori
        if self.precompute is None:
            adj=g.adj().to(self.device)
            adj=getNormAugAdj(adj,self.device).to(self.device)
            self.precompute=mulAdj(adj, self.K)
            if self.is_linear:
                a=(1-self.alpha)*feat+self.alpha*self.feat_ori
                self.precompute=  torch.sparse.mm(self.precompute,feat)  
            self.precompute=self.precompute.to(self.device)
        if (not self.is_linear):
            h=(1-self.alpha)*feat+self.alpha*self.feat_ori
            h=torch.sparse.mm(self.precompute,feat)
        else:
            h=self.precompute
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h


