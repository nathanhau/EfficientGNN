import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

def mulAdj(adj,K):
    for _ in range(K-1):
        adj=torch.sparse.mm(adj,adj)
    return adj
    
def getNormAugAdj(adj,aug=False):
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

