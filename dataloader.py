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
    data, split_idx = load_obgn()
    print(data)
    print(split_idx)