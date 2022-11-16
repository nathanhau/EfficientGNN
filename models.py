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
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, bias=True, norm=None, dropout=0, iso=False, adj=None):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.activation = activation
        self.bias = bias
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self.iso = iso
        self.adj = adj
        self.layers = nn.ModuleList()
        self.nm = nn.ModuleList()

        for i in range(self.n_layers):
            ind = self.n_hidden if i > 0 else self.in_feats
            outd = self.n_hidden if i < n_layers-1 else self.n_classes
            self.layers.append(conv.GraphConv(ind, outd, "both", self.bias))
            if i < self.n_layers-1:
                if self.norm == "bn":
                    self.nm.append(nn.BatchNorm1d(n_hidden))
                elif self.norm == "ln":
                    self.append(nn.LayerNorm(n_hidden))

        if self.iso and (self.adj != None):
            stdv = self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
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


class DeepLinear(nn.Module):
    def __init__(self, model, model_args, in_feats, n_hidden, n_classes, K, n_layers, activation, bias=True, norm=None, dropout=0, iso=False, adj=None):
        super(DeepLinear, self).__init__()
        self.model = model
        self.model_args = model_args
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.K = K
        self.activation = activation
        self.bias = bias
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self.iso = iso
        self.adj = adj
        self.prelayer = None
        self.layers = nn.ModuleList()
        self.klist = []
        # self.precompute=precompute

        if not isinstance(n_layers, list):
            self.klist = [K//n_layers for i in range(n_layers)]
            self.klist[0] += self.K % n_layers
            self.n_layers = n_layers
        else:
            self.klist = n_layers
            self.n_layers = len(n_layers)

        if self.n_layers > 1:
            if model == "SGC":
                self.layers.append(SGC(self.in_feats, self.n_hidden, self.klist[0], bias=self.bias, norm=self.norm))
                for i in range(1, self.n_layers-1):
                    self.layers.append(SGC(self.n_hidden, self.n_hidden, self.klist[i], bias=self.bias, norm=self.norm))
                self.layers.append(SGC(self.n_hidden, self.n_classes, self.klist[-1], bias=self.bias, norm=self.norm))
            elif model == "SGCRes":
                self.prelayer = nn.Linear(self.in_feats, self.n_hidden, bias=self.bias)
                for i in range(0, self.n_layers-1):
                    self.layers.append(SGCRes(self.n_hidden, self.n_hidden, self.klist[i], self.model_args["alpha"], bias=self.bias, norm=self.norm))
                self.layers.append(SGCRes(self.n_hidden, self.n_classes, self.klist[-1], self.model_args["alpha"], bias=self.bias, norm=self.norm))
            elif model == "SSGC":
                self.prelayer = nn.Linear(self.in_feats, self.n_hidden, bias=self.bias)
                for i in range(0, self.n_layers-1):
                    self.layers.append(SSGC(self.n_hidden, self.n_hidden, self.klist[i], self.model_args["alpha"], bias=self.bias, norm=self.norm))
                self.layers.append(SSGC(self.n_hidden, self.n_classes, self.klist[-1], self.model_args["alpha"], bias=self.bias, norm=self.norm))
            elif model == "DGC":
                self.layers.append(DGC(self.in_feats, self.n_hidden, self.klist[0], self.model_args["T"], bias=self.bias, norm=self.norm))
                for i in range(1, self.n_layers-1):
                    self.layers.append(DGC(self.n_hidden, self.n_hidden, self.klist[i], self.model_args["T"], bias=self.bias, norm=self.norm))
                self.layers.append(DGC(self.n_hidden, self.n_classes, self.klist[-1], self.model_args["T"], bias=self.bias, norm=self.norm))
            else:
                raise Exception("Model not implemented")
        else:
            raise Exception("Number of layers must be more than 1")

        if self.iso and (self.adj != None):
            stdv = self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
                    layer.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, features):
        h = features
        h = self.dropout(h)
        if self.model in ["SGCRes", "SSGC"]:
            h = self.prelayer(h)
            h = self.activation(h)
            h0 = h
            h = self.dropout(h)
        for i, layer in enumerate(self.layers):
            if self.model in ["SGCRes", "SSGC"]:
                h = layer(g, h, feat_ori=h0)
            else:
                h = layer(g, h)
            if i < len(self.layers)-1:
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


class LinearMLP(nn.Module):
    def __init__(self, model, model_args, in_feats, n_hidden, n_classes, K, mlp_d, activation, mlp_activation, bias=True, norm=None, dropout=0,):
        super(LinearMLP, self).__init__()
        self.model = model
        self.model_args = model_args
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.K = K
        self.n_mlp = len(mlp_d)+1
        self.bias = bias
        self.activation = activation
        self.mlp_activation = mlp_activation
        self.norm = norm
        self.mlp = nn.ModuleList()
        self.nm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.gnn = None
        if model == "SGC":
            self.gnn = SGC(self.in_feats, self.n_hidden, self.K, bias=self.bias, norm=self.norm)
        elif model == "SSGC":
            self.gnn = SSGC(self.in_feats, self.n_hidden, self.K, self.model_args["alpha"], bias=self.bias, norm=self.norm)
        elif model == "DGC":
            self.gnn = DGC(self.in_feats, self.n_hidden, K, self.model_args["T"], bias=self.bias, norm=self.norm)
        else:
            raise Exception("Model not implemented")
        prev = self.n_hidden
        for i in mlp_d:
            self.mlp.append(nn.Linear(prev, i, bias=self.bias))
            if self.norm == "bn":
                self.nm.append(nn.BatchNorm1d(i))
            elif self.norm == "ln":
                self.nm.append(nn.LayerNorm(i))
            prev = i
        self.mlp.append(nn.Linear(prev, n_classes, bias=self.bias))

    def forward(self, g, feat):
        h = self.dropout(feat)
        h = self.gnn(g, h)
        if self.activation is not None:
            h = self.activation(h)
        h = self.dropout(h)
        for i in range(self.n_mlp):
            h = self.mlp[i](h)
            if (i < self.n_mlp-1):
                if self.norm is not None:
                    h = self.nm[i](h)
                h = self.mlp_activation(h)
                h = self.dropout(h)
        return h

