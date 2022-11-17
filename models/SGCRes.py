import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter
from utilities.utils import mulAdj
from models.DGC import DGC

class SGCRes(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,device,bias=True,norm=None):
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
            adj=g.adj()
            self.precompute=mulAdj(adj, self.K)
        self.precompute=self.precompute.to(device)
        h=(1-self.alpha)*feat+self.alpha*self.feat_ori
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h


