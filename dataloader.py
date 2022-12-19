# from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
import torch
import dgl
from time import perf_counter
from utilities.utils import getNormAugAdj,mulAdj,mulAdj2
import numpy as np

def load_obgn(dataset="obgn-arxiv"):
    dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    graph,label=dataset[0]
    split_idx= dataset.get_idx_split()
    return graph,label,split_idx

def load_data(dataset):
    graph=None
    label=None
    split_idx=None
    if dataset=="arxiv":
        ds=DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
        graph,label=ds[0]
        label=label.squeeze()
        split_idx= ds.get_idx_split()
        maskdict={}
        for (k,v) in split_idx.items():
            mask=[False]*graph.num_nodes()
            for i in v:
                mask[i]=True
            if k=="valid":
                maskdict['val_mask']=torch.tensor(mask)
            else:
                maskdict[k+'_mask']=torch.tensor(mask)
        return graph,label,maskdict    

    else:

        if dataset=="cora":
            ds=CoraGraphDataset(raw_dir="dataset/")
        elif dataset=="citeseer":
            ds=CiteseerGraphDataset(raw_dir="dataset/")
        elif dataset=="pubmed":
            ds=PubmedGraphDataset(raw_dir="dataset/",reverse_edge=False)
        else:
            raise ValueError("Unrecognized dataset (\"arxiv/cora/citeseer/pubmed\")")
        graph=ds[0]
        label=graph.ndata['label']
        split_idx={}
        # for s in ['train','val','test']:
        #     print("Converting "+s+" mask" )
        #     idx=[]
        #     for i,x in enumerate(graph.ndata[s+'_mask'].tolist()):
        #         if x:
        #             idx.append(i)
        #     if s=='val':
        #         split_idx['valid']=torch.tensor(idx)
        #     else:
        #         split_idx[s]=torch.tensor(idx)
        split_idx['train_mask']=graph.ndata['train_mask']
        split_idx['val_mask']=graph.ndata['val_mask']
        split_idx['test_mask']=graph.ndata['test_mask']
    return graph,label,split_idx

def preprocess_graph(g,self_loop=True):
    g=dgl.to_bidirected(g,copy_ndata=True);
    if self_loop:
        g.remove_self_loop()
        g.add_self_loop()
    return g


def get_prep_ogbn(dataset):
    data = DglNodePropPredDataset(name = dataset, root = 'dataset/')
    g, labels = data[0]
    labels = labels[:, 0]
    g.ndata['label'] = labels
    g = dgl.add_reverse_edges(g)
    features = g.ndata['feat']
    idx_split = data.get_idx_split()
    train_mask = idx_split['train']
    val_mask = idx_split['valid']
    test_mask = idx_split['test']
    in_feats = features.shape[1]
    n_classes = (labels.max() + 1).item()
    return g, features, labels, n_classes, in_feats, train_mask, val_mask, test_mask

def get_prep_pubmed():
    data = PubmedGraphDataset()
    g = data[0]
    g = g.to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[-1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, features, labels, n_classes, in_feats, train_mask, val_mask, test_mask

if __name__=="__main__":
    graph, label, split_idx = load_data('cora')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
    # graph = preprocess_graph(graph).to(device)
    # # print(graph.adj().to_dense()[0][0])
    # adj=graph.adj()
    # print((adj.transpose(0,1)==adj).all())
    adj=torch.randint(2,(1000,1000)).float().to_sparse()
    t1=perf_counter()
    a=getNormAugAdj(adj)
    t1=perf_counter()-t1
    # a=adj.to_dense()
    # t2=perf_counter()
    # b=normalized_adjacency(a)
    # t2=perf_counter()-t2
    print(t1)
    # print(t2)
    # print(a.to_dense())
    # print(b)
    # print(np.array_equal(a.to_dense().numpy(),b))
    # g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([2, 1, 0])))
    # print(getNormAugAdj(g.adj()).to_dense() )
    # g.add_self_loop()
    # print(g.adj().to_dense())
    a1=torch.randint(1,6,(50,50)).float()
    # a1=torch.tensor([[2,3,5,2],[3,4,2,5],[4,5,4,3],[5,5,1,4]]).float()
    # print(a1)
    a1=a1.to_sparse()
    # a2=torch.rand(1000,1000)
    t3=perf_counter()
    # a3=torch.sum(a1,1)
    print(mulAdj(a1, 10).to_dense())
    print(perf_counter()-t3)
    # a1=a1.to_sparse()
    # a2=a2.to_sparse()
    t3=perf_counter()
    print(mulAdj2(a1,10).to_dense())
    print(perf_counter()-t3)
    