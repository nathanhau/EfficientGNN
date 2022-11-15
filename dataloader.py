from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
import torch
import dgl

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
        split_idx= ds.get_idx_split()
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
        for s in ['train','val','test']:
            print("Converting "+s+" mask" )
            idx=[]
            for i,x in enumerate(graph.ndata[s+'_mask'].tolist()):
                if x:
                    idx.append(i)
            if s=='val':
                split_idx['valid']=torch.tensor(idx)
            else:
                split_idx[s]=torch.tensor(idx)
    return graph,label,split_idx

def preprocess_graph(g,self_loop=True):
    g=dgl.to_bidirected(g,copy_ndata=True);
    if self_loop:
        g.remove_self_loop()
        g.add_self_loop()
    return g


if __name__=="__main__":
    data, split_idx = load_obgn()
    print(data)
    print(split_idx)