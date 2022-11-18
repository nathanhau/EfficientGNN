import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

class SSGC(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,device,
                bias=True,norm=None):
        super(SSGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.alpha=alpha
        self.feat_ori=None
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.reset_parameters()
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
        feat=feat.to(self.device)
        self.feat_ori=self.feat_ori.to(self.device)
        adj=g.adj().to(self.device)
        h = torch.zeros_like(feat).to(self.device)
        for i in range(self.K):
            feat = torch.spmm(adj, feat)
            h += (1-self.alpha)*feat + self.alpha*self.feat_ori
            h /= self.K
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h

def ssgc_precompute(features,adj,K,alpha,device):
    t=perf_counter()
    features=features.to(device)
    adj=adj.to(device)
    feat_ori=features
    feature_set = torch.zeros_like(features)
    for i in range(K):
        features = torch.spmm(adj, features)
        feature_set += (1-alpha)*features + alpha*feat_ori
    feature_set /= K
    precompute_time = perf_counter()-t
    return feature_set, precompute_time
