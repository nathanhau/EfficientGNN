import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import citegrh
import dgl

# load and preprocess the pubmed dataset
data = citegrh.load_pubmed()

# sparse bag-of-words features of papers
features = torch.FloatTensor(data.features)
print('features',features)
# the number of input node features
in_feats = features.shape[1]
print('in_feats',in_feats)
# class labels of papers
labels = torch.LongTensor(data.labels)
print('labels',labels)
# the number of unique classes on the nodes.
n_classes = data.num_labels
print('n_classes',n_classes)

graph = dgl.remove_self_loop(data[0])
print('graph',graph)