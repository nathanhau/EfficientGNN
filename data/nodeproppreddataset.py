from ogb.nodeproppred import NodePropPredDataset


dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')

print('Number of categories:', dataset.num_classes)
print('Number of graphs in this dataset:', len(dataset))

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
print("train_idx", train_idx)
print("valid_idx", valid_idx)
print("test_idx", test_idx)

graph, label = dataset[0]
print("graph", graph)
print("label", label)

edge_index = graph['edge_index']
print("edge_index", edge_index)
edge_feat = graph['edge_feat']
print("edge_feat", edge_feat)
node_feat = graph['node_feat']
print("node_feat", node_feat)
num_nodes = graph['num_nodes']
print("num_nodes", num_nodes)

n_classes = dataset.num_classes
print("n_classes", n_classes)
in_feats = node_feat.shape[1]
print("in_feats", in_feats)
