from trainer import preprocess_linear, train_linear, test, train
from dataloader import load_obgn, load_data, preprocess_graph, get_prep_ogbn, get_prep_pubmed

from models.SGCRes import SGCRes
from models.SSGC import SSGC
from models.DGC import DGC
from models.SGC import SGC
from stacked_models import DeepLinear, LinearMLP
from models.GCN import GCN



from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl
from time import perf_counter


#hyperparam 
device = 'cpu'
activation = nn.ReLU()
epochs = 50
batch_size = 10000
lr = 0.02
loss_fn = nn.CrossEntropyLoss()
weight_decay = 5e-4

# DGC
bias = False

def train(model, g, features, labels, train_mask, val_mask, test_mask):
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(g, features)
        train_acc = torch.sum(logits[train_mask].argmax(1) == labels[train_mask]).item() / train_mask.sum().item()
        val_acc = torch.sum(logits[val_mask].argmax(1) == labels[val_mask]).item() / val_mask.sum().item()
        test_acc = torch.sum(logits[test_mask].argmax(1) == labels[test_mask]).item() / test_mask.sum().item()
        print(f'Epoch {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')

    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')

def test(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        print(f'Test: {acc:.4f}')
        return acc



if __name__ == "__main__":
  import argparse 

  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset", help="arxiv|cora|citeseer|pubmed")
  parser.add_argument("--model", help="sgc|ssgc|dgc|deeplinear|linearmlp|gcn|sgcres")

  args = parser.parse_args()

  graph, label, split_idx = load_data(args.dataset)

  graph = preprocess_graph(graph)

  raw_features = graph.ndata['feat']

  # alpha is used for ssgc

  precomputed,pt = preprocess_linear(graph, raw_features , args.model, K=2, T=2, alpha=None)

  in_feats = precomputed.shape[-1]
  n_classes = len(label)


  if args.model == 'sgc': 
    model = SGC(in_feats, n_classes, 2, True, None)
  elif args.model == 'ssgc':
    model = SSGC(in_feats, n_classes, K=2, alpha=0.1)
  elif args.model == 'dgc': 
    model = DGC(in_feats, n_classes, K=2, T=2, bias=bias)
  elif args.model == 'deeplinear':
    model = DeepLinear("SGC",{},in_feats, 16, n_classes, 2, 2, activation, dropout=0.2)
  elif args.model == 'linearmlp':
    model = LinearMLP("SGC",{},in_feats, 16, n_classes, 2, 2, activation, dropout=0.2)
  elif args.model == 'gcn':
    model = GCN(in_feats, 16, n_classes, 2, activation, dropout=0.2)


  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  
  train(model, graph, precomputed, label, split_idx["train"], split_idx["valid"], split_idx["test"])
  test(model, graph, precomputed, label, split_idx["test"])
