import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

class DeepSGC(nn.Module):
    def __init__(self,in_feats,n_hidden,n_classes,k,n_layers,activation,bias=True,norm=False,dropout=0,iso=False,adj=None,rewiring=False):
        super(DeepSGC,self).__init__()
        self.in_feats=in_feats
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.k=k
        self.activation=activation
        self.bias=bias
        self.norm=norm,
        self.dropout=nn.Dropout(p=dropout)
        self.iso=iso
        self.adj=adj
        self.layers=nn.ModuleList()
        self.klist=[]
        if not isinstance(n_layers, list):
            self.klist=[k//n_layers for i in range(n_layers)]
            self.klist[0]+=k%n_layers
            self.n_layers=n_layers
        else:
            self.klist=n_layers
            self.n_layers=len(n_layers)

        if self.n_layers>1:
            self.layers.append(conv.SGConv(in_feats,n_hidden,self.klist[0],True,bias=bias,norm=norm))
            for i in range(1,n_layers-1):
                self.layers.append(conv.SGConv(n_hidden,n_hidden,self.klist[i],True,bias=bias,norm=norm))
            self.layers.append(conv.SGConv(n_hidden,n_classes,self.klist[-1],True,bias=bias,norm=norm))
        else:
            self.layers.append(conv.SGConv(in_feats,n_hidden,self.klist[0],True,bias=bias,norm=norm))
        if self.iso and (self.adj != None):
            stdv=self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
                    layer.fc.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.fc.bias.data.uniform_(-stdv,stdv)
        
    def forward(self,g,features):
        h=features
        for i,layer in enumerate(self.layers):
            h=layer(g,h)
            if i<self.layers.len()-1:
                h=self.activation(h)
                h=self.dropout(h)
        return h

    def get_std(self):
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim = 1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim = 0).to_dense().unsqueeze(0) + 1
        # support = torch.sqrt(torch.sum(d_i * d_j))
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        # print(stdv)
        return stdv

#rewrite to fit all linear layers
class DeepLinear(nn.Module):
    def __init__(self,model,in_feats,n_hidden,n_classes,k,n_layers,precompute,activation,bias=True,norm=False,dropout=0,iso=False,adj=None,rewiring=False):
        super(DeepLinear,self).__init__()
        self.model=model
        self.in_feats=in_feats
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.k=k
        self.activation=activation
        self.bias=bias
        self.norm=norm,
        self.dropout=nn.Dropout(p=dropout)
        self.iso=iso
        self.adj=adj
        self.layers=nn.ModuleList()
        self.klist=[]
        self.precompute=precompute

        if not isinstance(n_layers, list):
            self.klist=[k//n_layers for i in range(n_layers)]
            self.klist[0]+=k%n_layers
            self.n_layers=n_layers
        else:
            self.klist=n_layers
            self.n_layers=len(n_layers)

        if self.n_layers>1:
            self.layers.append(nn.Linear(in_feats,n_hidden,bias=self.bias))
            for i in range(1,n_layers-1):
                self.layers.append(nn.Linear(n_hidden,n_hidden,bias=self.bias))
            self.layers.append(nn.Linear(n_hidden,n_classes,bias=self.bias))
        else:
            self.layers.append(nn.Linear(in_feats,n_classes,bias=self.bias))
        if self.iso and (self.adj != None):
            stdv=self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
                    layer.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.bias.data.uniform_(-stdv,stdv)
        
    def forward(self,g,features):
        h=features
        for i,layer in enumerate(self.layers):
            h=layer(g,h)
            if i<self.layers.len()-1:
                h=self.activation(h)
                h=self.dropout(h)
        return h

    def get_std(self):
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim = 1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim = 0).to_dense().unsqueeze(0) + 1
        # support = torch.sqrt(torch.sum(d_i * d_j))
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        # print(stdv)
        return stdv


class SGC(nn.Module):
    def __init__(self,in_feats,n_classes,bias=True):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None

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
        return h

class SGCRes(nn.Module):
    def __init__(self,in_feats,n_classes,bias=True):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.feat_ori=None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self,g,feat):
        if self.feat_ori is None:
            self.feat_ori=feat
        if self.precompute is None:
            adj=g.adj()
            self.precompute=mulAdj(adj, self.K)
        h=(1-self.alpha)*feat+self.alpha*self.feat_ori
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        return h

class SSGC(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,
                bias=True):
        super(SSGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.alpha=alpha
        self.feat_ori=None
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self,g,feat):
        if self.feat_ori is None:
            self.feat_ori=feat
        adj=g.adj()
        h = torch.zeros_like(feat)
        for i in range(K):
            feat = torch.spmm(adj, feat)
            h += (1-self.alpha)*feat + self.alpha*self.feat_ori
            h /= self.K
        h=self.fc(h)
        return h

class DGC(nn.Module):
    def __init__(self,in_feats,n_classes,T,K,bias=True):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.T=T
        self.K=K
        self.delta=T/K
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self,g,feat):
        if self.precompute is None:
            adj=g.adj()
            S=getNormAugAdj(adj,aug=True)
            I=torch.eye(adj.shape[0]).to_sparse()
            self.precompute=(1-self.delta)*I + self.delta*S
            for _ in range(self.K-1):
                self.precompute=torch.sparse.mm(self.precompute,self.precompute)
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        return h


def ssgc_precompute(features,adj,K,alpha):
    t=perf_counter()
    feat_ori=features
    feature_set = torch.zeros_like(features)
    for i in range(K):
        features = torch.spmm(adj, features)
        feature_set += (1-alpha)*features + alpha*feat_ori
    feature_set /= K
    precompute_time = perf_counter()-t
    return feature_set, precompute_time

#Input adj. matrix without self-loop
def dgc_precompute(features,adj,T,K):
    print(features)
    if T==0 or K==0:
        return feature,0
    delta=T/K
    t=perf_counter()
    S=getNormAugAdj(adj,aug=True)
    I=torch.eye(adj.shape[0]).to_sparse()
    S_delta=(1-delta)*I + delta*S
    for _ in range(K):
        features=torch.sparse.mm(S_delta,features)
    precompute_time=perf_counter()-t
    print(features,precompute_time)
    return features,precompute_time

def mulAdj(adj,K):
    for _ in range(self.K-1):
        adj=torch.sparse.mm(adj,adj)
    return adj
    
def getNormAugAdj(adj,aug=True):
    # print(adj.to_dense())
    if aug:
        adj=adj+torch.eye(adj.shape[0]).to_sparse()
    # print(adj.to_dense())
    d=torch.sparse.sum(adj,1)
    # print(d.to_dense())
    d=torch.pow(d,-0.5).to_dense()
    # print(d)
    d=torch.diag(d).to_sparse()
    support=torch.sparse.mm(d,adj)
    return torch.sparse.mm(support,d)


g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
g=dgl.to_bidirected(g)
# g = dgl.add_self_loop(g)
# print(getNormAugAdj(g.adj()).to_dense())
feat=torch.rand(6,5)
ssgc_precompute(feat, g.adj(), 5, 0.2)
a=DGC(5,3,5,3)
out=a(g,feat)
# print(out)
# model=DeepSGC(10, 6, 3, 7, 3, F.relu,iso=True,adj=g.adj())
# for i in model.layers:
#     print(i.fc.weight.shape)
#     print(i._k)