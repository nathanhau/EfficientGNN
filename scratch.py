import torch
import dgl
from dgl.nn.pytorch import conv
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
from dataloader import load_data,load_obgn
import pickle
import argparse

# print(torch.cuda.is_available())
# args=argparse.ArgumentParser()

# args.add_argument("--T",type=int,default=None)
# args=args.parse_args()

# print(args.T)

# with open("citeseer.txt", 'rb') as f:
#             a= pickle.load(f)['weight_decay']
#             print(a)
# # g= dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
a,b,c=load_data('cora')
print(a.ndata['feat'])
# conv=conv.GraphConv(3,6,'both',bias=True)
# print(b.shape)
# d,e,f=load_data('arxiv')
# print(e.shape)
dataset = CoraGraphDataset()

graph = dataset[0]

feat = graph.ndata['feat']
torch.set_printoptions(threshold=10000)
d=torch.rand(4,4)
print(d is d.to(torch.device('cuda')))