from ogb.nodeproppred import DglNodePropPredDataset

def load_obgn(dataset="obgn-arxiv"):
    dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    data=dataset[0]
    split_idx= dataset.get_idx_split()
    return data, split_idx

if __name__=="__main__":
    data, split_idx = load_obgn()
    print(data)
    print(split_idx)