import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np
import scipy.sparse as sp
from time import perf_counter

def mulAdj2(adj,K):
    nadj=adj
    for _ in range(K-1):
        nadj=torch.sparse.mm(nadj,adj)
    return nadj
    
def mulAdj(adj,K):
    a=adj.to_dense().numpy()
    a=np.linalg.matrix_power(a, K)
    return torch.FloatTensor(a).to_sparse()    
def getNormAugAdj(adj,aug=False):
    # print(adj.to_dense())
    adj=adj.to_dense()+torch.eye(adj.shape[0])
    # print(adj.to_dense())
    d=torch.sum(adj,1)
    # print(d.to_dense())
    d=torch.pow(d,-0.5).flatten()
    d=torch.diag(d).to_sparse()
    # print(d.to_dense())
    support=torch.sparse.mm(d,adj.to_sparse())
    # print(support.to_dense())
    return torch.sparse.mm(support,d)

def getNormAugAdj2(adj):
   adj =  torch.eye(adj.shape[0])+adj
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
#    print(row_sum)
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#    print(d_mat_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

# if __name__=="__main__":
    