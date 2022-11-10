from ogb.nodeproppred import DglNodePropPredDataset

device = None
dataset = DglNodePropPredDataset(name="ogbn-arxiv", root='dataset/')

print('Number of categories:', dataset.num_classes)
print('Number of graphs in this dataset:', len(dataset))

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

print("graph", graph)
print("labels", labels)

srcs, dsts = graph.all_edges()
print("srcs", srcs)
print("dsts", dsts)

in_feats = graph.ndata["feat"].shape[1]
print("in_feats", in_feats)
n_classes = (labels.max() + 1).item()
print("n_classes", n_classes)

train_idx = train_idx.to(device)
val_idx = valid_idx.to(device)
test_idx = test_idx.to(device)
labels = labels.to(device)
graph = graph.to(device)

print(train_idx)
print(val_idx)
print(test_idx)
print(labels)
print(graph)