import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.nn.pytorch.conv import SGConv

lr=0.2
bias=False
n_epochs=100
weight_decay=5e-4

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask] # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

data = PubmedGraphDataset()
g = data[0]
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = g.number_of_edges()
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
model = SGConv(in_feats,
                n_classes,
                k=2,
                cached=True,
                bias=bias)
loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
dur = []
for epoch in range(n_epochs):
    model.train()
    if epoch >= 3:
        t0 = time.time()
    # forward
    logits = model(g, features) # only compute the train set
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(model, g, features, labels, val_mask)
    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

print()
acc = evaluate(model, g, features, labels, test_mask)
print("Test Accuracy {:.4f}".format(acc))