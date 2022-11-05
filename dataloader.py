from ogb.nodeproppred import DglNodePropPredDataset

def load_obgn(dataset="obgn-arxiv"):
    dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    data=dataset[0]
    split_idx= dataset.get_idx_split()
    return data, split_idx