import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

def mulAdj(adj,K):
    nadj=adj
    for _ in range(K-1):
        nadj=torch.sparse.mm(nadj,adj)
    return nadj
    
def mulAdj2(adj,K):
    adj=sparsetoscipy(adj)
    # a=adj.to_dense().numpy()
    # print(adj)
    a=adj**K
    a=a.tocoo()
    # print(a)
    # print(type(a))
    return sparsetopytorch(a)    
def getNormAugAdj1(adj,aug=False):
    # print(adj.is_sparse)
    adj=adj.coalesce()
    values=adj.values().numpy()
    indices=adj.indices.numpy()
    adj=sp.coo_matrix(values,indices)
    b=sp.identity(adj.size(0),format='coo')
    # print(b)
    # values = b.data
    # indices = np.vstack((b.row, b.col))

    # i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)
    # shape = b.shape

    # b=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    adj=adj+b
    # print(adj.to_dense())
    d=torch.sparse.sum(adj,1)
    # print(d.to_dense())
    d=torch.pow(d,-0.5).flatten()
    d=torch.diag(d).to_sparse()
    # print(d.to_dense())
    support=torch.sparse.mm(d,adj)
    # print(support.to_dense())
    return torch.sparse.mm(support,d)

def getNormAugAdj2(adj):
    adj=sparsetoscipy(adj)
    # print(adj.shape)
    adj = sp.eye(adj.shape[0])+adj
    # adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
#    print(row_sum)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#    print(d_mat_inv_sqrt)
    s=d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return sparsetopytorch(s)

def getNormAugAdj(adj,device):
    # print(adj.to_dense())
    size=adj.shape[0]
    v=torch.ones(size)
    i=[[i for i in range(size)]]*2
    eye=torch.sparse_coo_tensor(i,v,(size,size)).to(device)
    adj=adj+eye
    # print(adj.to_dense())
    d=torch.sparse.sum(adj,1)
    # print(d.to_dense())
    d=torch.pow(d,-0.5).flatten().to_dense().tolist()
    # print(d)
    d=torch.sparse_coo_tensor(i,d,(size,size)).to(device)
    # print(d.to_dense())
    support=torch.sparse.mm(d,adj)
    # print(support.to_dense())
    return torch.sparse.mm(support,d)

def sparsetoscipy(mat):
    mat=mat.coalesce()
    values=mat.values().numpy()
    indices=mat.indices().int().numpy()
    # print(indices[0].shape)
    mat=sp.coo_matrix((values,(indices[0],indices[1])),shape=(mat.size(1),mat.size(1)))
    return mat

def sparsetopytorch(mat):
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
# if __name__=="__main__":
    