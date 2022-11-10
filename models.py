import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

class GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,n_classes,n_layers,activation,bias=True,norm=None,dropout=0,iso=False,adj=None):
        super(GCN,self).__init__()
        self.in_feats=in_feats
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.n_layers=n_layers
        self.activation=activation
        self.bias=bias
        self.norm=norm
        self.dropout=nn.Dropout(p=dropout)
        self.iso=iso
        self.adj=adj
        self.layers=nn.ModuleList()
        self.nm=nn.ModuleList()

        for i in range(self.n_layers):
            ind=self.n_hidden if i>0 else self.in_feats
            outd=self.n_hidden if i<n_layers-1 else self.n_classes
            self.layers.append(conv.GraphConv(ind,outd,"both",self.bias))
            if i<self.n_layers-1:
                if self.norm=="bn":
                    self.nm.append(nn.BatchNorm1d(n_hidden))
                elif self.norm=="ln":
                    self.append(nn.LayerNorm(n_hidden))

        if self.iso and (self.adj != None):
            stdv=self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
                    layer.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.bias.data.uniform_(-stdv,stdv)
    
    def forward(self,g,feat):
        h=feat
        h=self.dropout(h)
        for i in range(self.n_layers):
            h=self.layers[i](g,h)
            if i<self.n_layers-1:
                if self.norm is not None:
                    h=self.nm[i](h)
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
    def __init__(self,model,model_args,in_feats,n_hidden,n_classes,K,n_layers,activation,bias=True,norm=None,dropout=0,iso=False,adj=None):
        super(DeepLinear,self).__init__()
        self.model=model
        self.model_args=model_args
        self.in_feats=in_feats
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.K=K
        self.activation=activation
        self.bias=bias
        self.norm=norm
        self.dropout=nn.Dropout(p=dropout)
        self.iso=iso
        self.adj=adj
        self.prelayer=None
        self.layers=nn.ModuleList()
        self.klist=[]
        # self.precompute=precompute

        if not isinstance(n_layers, list):
            self.klist=[K//n_layers for i in range(n_layers)]
            self.klist[0]+=self.K%n_layers
            self.n_layers=n_layers
        else:
            self.klist=n_layers
            self.n_layers=len(n_layers)

        if self.n_layers>1:
            if model=="SGC":
                self.layers.append(SGC(self.in_feats,self.n_hidden,self.klist[0],bias=self.bias,norm=self.norm))
                for i in range(1,self.n_layers-1):
                    self.layers.append(SGC(self.n_hidden,self.n_hidden,self.klist[i],bias=self.bias,norm=self.norm))
                self.layers.append(SGC(self.n_hidden,self.n_classes,self.klist[-1],bias=self.bias,norm=self.norm))
            elif model=="SGCRes":
                self.prelayer=nn.Linear(self.in_feats,self.n_hidden,bias=self.bias)
                for i in range(0,self.n_layers-1):
                    self.layers.append(SGCRes(self.n_hidden,self.n_hidden,self.klist[i],self.model_args["alpha"],bias=self.bias,norm=self.norm))
                self.layers.append(SGCRes(self.n_hidden,self.n_classes,self.klist[-1],self.model_args["alpha"],bias=self.bias,norm=self.norm))
            elif model=="SSGC":
                self.prelayer=nn.Linear(self.in_feats,self.n_hidden,bias=self.bias)
                for i in range(0,self.n_layers-1):
                    self.layers.append(SSGC(self.n_hidden,self.n_hidden,self.klist[i],self.model_args["alpha"],bias=self.bias,norm=self.norm))
                self.layers.append(SSGC(self.n_hidden,self.n_classes,self.klist[-1],self.model_args["alpha"],bias=self.bias,norm=self.norm))
            elif model=="DGC":
                self.layers.append(DGC(self.in_feats,self.n_hidden,self.klist[0],self.model_args["T"],bias=self.bias,norm=self.norm))
                for i in range(1,self.n_layers-1):
                    self.layers.append(DGC(self.n_hidden,self.n_hidden,self.klist[i],self.model_args["T"],bias=self.bias,norm=self.norm))
                self.layers.append(DGC(self.n_hidden,self.n_classes,self.klist[-1],self.model_args["T"],bias=self.bias,norm=self.norm))
            else:
                raise Exception("Model not implemented")
        else:
            raise Exception("Number of layers must be more than 1")

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
        h=self.dropout(h)
        if self.model in ["SGCRes","SSGC"]:
            h=self.prelayer(h)
            h=self.activation(h)
            h0=h
            h=self.dropout(h)
        for i,layer in enumerate(self.layers):
            if self.model in ["SGCRes","SSGC"]:
                h=layer(g,h,feat_ori=h0)
            else:
                h=layer(g,h)
            if i<len(self.layers)-1:
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

class LinearMLP(nn.Module):
    def __init__(self,model,model_args,in_feats,n_hidden,n_classes,K,mlp_d,activation,mlp_activation,bias=True,norm=None,dropout=0,):
        super(LinearMLP,self).__init__()
        self.model=model
        self.model_args=model_args
        self.in_feats=in_feats
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.K=K
        self.n_mlp=len(mlp_d)+1
        self.bias=bias
        self.activation=activation
        self.mlp_activation=mlp_activation
        self.norm=norm
        self.mlp=nn.ModuleList()
        self.nm=nn.ModuleList()
        self.dropout=nn.Dropout(p=dropout)
        self.gnn=None
        if model=="SGC":
            self.gnn=SGC(self.in_feats,self.n_hidden,self.K,bias=self.bias,norm=self.norm)
        elif model=="SSGC":
            self.gnn=SSGC(self.in_feats, self.n_hidden, self.K, self.model_args["alpha"],bias=self.bias,norm=self.norm)
        elif model=="DGC":
            self.gnn=DGC(self.in_feats,self.n_hidden,K,self.model_args["T"],bias=self.bias,norm=self.norm)
        else:
            raise Exception("Model not implemented")
        prev=self.n_hidden
        for i in mlp_d:
            self.mlp.append(nn.Linear(prev,i,bias=self.bias))
            if self.norm=="bn":
                self.nm.append(nn.BatchNorm1d(i))
            elif self.norm=="ln":
                self.nm.append(nn.LayerNorm(i))
            prev=i
        self.mlp.append(nn.Linear(prev,n_classes,bias=self.bias))

    def forward(self,g,feat):
        h=self.dropout(feat)
        h=self.gnn(g,h)
        if self.activation is not None:
            h=self.activation(h)
        h=self.dropout(h)
        for i in range(self.n_mlp):
            h=self.mlp[i](h)
            if (i<self.n_mlp-1):
                if self.norm is not None:
                    h=self.nm[i](h)
                h=self.mlp_activation(h)
                h=self.dropout(h)
        return h

            
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

class SGCRes(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,bias=True,norm=None):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.K=K
        self.alpha=alpha
        self.fc=nn.Linear(in_feats,n_classes,bias=bias)
        self.precompute=None
        self.feat_ori=None
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

    def forward(self,g,feat,feat_ori=None):
        if self.feat_ori is None:
            if feat_ori is None:
                self.feat_ori=feat
            else:
                self.feat_ori=feat_ori
        if self.precompute is None:
            adj=g.adj()
            self.precompute=mulAdj(adj, self.K)
        h=(1-self.alpha)*feat+self.alpha*self.feat_ori
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h

class SSGC(nn.Module):
    def __init__(self,in_feats,n_classes,K,alpha,
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
        adj=g.adj()
        h = torch.zeros_like(feat)
        for i in range(self.K):
            feat = torch.spmm(adj, feat)
            h += (1-self.alpha)*feat + self.alpha*self.feat_ori
            h /= self.K
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h

class DGC(nn.Module):
    def __init__(self,in_feats,n_classes,K,T,bias=True,norm=None):
        super(DGC,self).__init__()
        self.in_feats=in_feats
        self.n_classes=n_classes
        self.T=T
        self.K=K
        self.delta=T/K
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
            S=getNormAugAdj(adj,aug=True)
            I=torch.eye(adj.shape[0]).to_sparse()
            self.precompute=(1-self.delta)*I + self.delta*S
            for _ in range(self.K-1):
                self.precompute=torch.sparse.mm(self.precompute,self.precompute)
        h=torch.sparse.mm(self.precompute,feat)
        h=self.fc(h)
        if self.norm is not None:
            h=self.nm(h)
        return h

def sgc_precompute(features,adj,K):
    t=perf_counter()
    for i in range(K):
        features=torch.spmm(adj,features)
    precompute_time=perf_counter()-t
    return features,precompute_time


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
    for _ in range(K-1):
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

