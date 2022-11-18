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


# hyperparam
device = 'cpu'
activation = nn.ReLU()
epochs = 50
batch_size = 10000
lr = 0.02
loss_fn = nn.CrossEntropyLoss()
weight_decay = 5e-4

# DGC

def train(model, g, features, labels, train_mask, val_mask,lr,weight_decay,epochs,is_linear=False):
    t = perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask=train_mask.to(device)
    val_mask=val_mask.to(device)
    labels=labels.to(device)
    g=g.to(device)
    features=features.to(device)
    for epoch in range(epochs):
        model.train()
        logits = None
        if is_linear:
            logits = model(features)
        else:
            logits = model(g, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        model.eval()
        if is_linear:
            logits = model(features)
        else:
            logits = model(g, features)
        train_acc = torch.sum(logits[train_mask].argmax(1) == labels[train_mask]).item() / train_mask.sum().item()
        val_acc = torch.sum(logits[val_mask].argmax(1) == labels[val_mask]).item() / val_mask.sum().item()
        # test_acc = torch.sum(logits[test_mask].argmax(1) == labels[test_mask]).item() / test_mask.sum().item()
        print(f'Epoch {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')

    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')

def test(model, g, features, labels, mask,is_linear=False):
    model.eval()
    mask=mask.to(device)
    with torch.no_grad():
        logits = None
        if is_linear:
            logits = model(features)[mask]
        else:
            logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        print(f'Test: {acc:.4f}')
        return acc


K = 2
lr = 0.01
wd = 5e-4
T = 1
alpha = 0.1
dropout = 0.5
k_hop = 5
num_layers = 5

if __name__ == "__main__":
  import argparse,gc
  gc.collect()
  del variables
  torch.cuda.empty_cache()
  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset", help="arxiv|cora|citeseer|pubmed")
  parser.add_argument("--mode", type=str, choices=["linear","linearmlp","deeplinear"])
  parser.add_argument("--model", type=str, choices=["sgc","ssgc","dgc","sgc_res","gcn"], help="sgc|ssgc|dgc|sgc_res|gcn")
  parser.add_argument("--K","--k", type=int,default=2)
  parser.add_argument("--lr",type=float)
  parser.add_argument("--wd",type=str)
  parser.add_argument("--T",type=float)
  parser.add_argument("--alpha",type=float)
  parser.add_argument("--epochs",type=int)
  parser.add_argument("--layer_k",nargs="+",type=int)
  parser.add_argument("--deep_hidden_d",type=int,default=32)
  parser.add_argument("--dropout",type=float,default=0)
  parser.add_argument("--norm",type=str,choices=["bn","ln"])
  parser.add_argument("--iso",action='store_true')
  parser.add_argument("--no-bias",action='store_false')


  args=parser.parse_args()
  graph, label, split_idx = load_data(args.dataset)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  graph = preprocess_graph(graph).to(device)
  label=label.to(device)
  # print(graph.ndata['feat'])
  raw_features = graph.ndata['feat'].to(device)

  in_feats = raw_features.shape[-1]
  n_classes = label.max().item()+1

  # alpha is used for ssgc
  if args.mode=="linear":
    # precomputed,pt = preprocess_linear(graph, raw_features , args.model, args.K, device,T=args.T, alpha=args.alpha)
    # ln=nn.Linear(raw_features.shape[1],n_classes).to(device)
    # train(ln,graph,precomputed,label,split_idx['train_mask'],split_idx['val_mask'],args.lr,float(args.wd),args.epochs,is_linear=True)
    # test(ln,graph,precomputed,label,split_idx['test_mask'],is_linear=True)

    if args.model == 'sgc': 
      model = SGC(in_feats, n_classes, args.K, device)
    elif args.model == 'ssgc':
      model = SSGC(in_feats, n_classes, args.K, args.alpha, device)
    elif args.model == 'dgc': 
      model = DGC(in_feats, n_classes, args.K, args.T, device)
    model=model.to(device)
    train(model, graph, raw_features, label, split_idx["train_mask"], split_idx["val_mask"],args.lr,float(args.wd),args.epochs)
    test(model, graph, raw_features, label, split_idx["test_mask"])
  elif args.mode=="deeplinear":

    adj=graph.adj() if args.iso else None
    if args.model == 'sgc': 
      model = DeepLinear("SGC",{},in_feats, args.deep_hidden_d, n_classes, args.K, args.layer_k, activation,device,
               args.no_bias,args.norm,args.dropout,args.iso,adj)
    elif args.model == 'ssgc':
      model = DeepLinear("SSGC",{"alpha":args.alpha},in_feats, args.deep_hidden_d, n_classes, args.K, args.layer_k, activation,device,
               args.no_bias,args.norm,args.dropout,args.iso,adj)
    elif args.model == 'dgc': 
      model = DeepLinear("DGC",{"T":args.T},in_feats, args.deep_hidden_d, n_classes, args.K, args.layer_k, activation,device,
               args.no_bias,args.norm,args.dropout,args.iso,adj)
    elif args.model == 'sgc_res':
      model = DeepLinear("SGCRes",{"alpha":args.alpha},in_feats, args.deep_hidden_d, n_classes, args.K, args.layer_k, activation,device,
               args.no_bias,args.norm,args.dropout,args.iso,adj)
    elif args.model == 'gcn':
      model = GCN(in_feats, args.deep_hidden_d, n_classes, args.layer_k, activation,device,
               args.no_bias,args.norm,args.dropout,args.iso,adj)
    model=model.to(device)

    # print(raw_features)
    train(model, graph, raw_features, label, split_idx["train_mask"], split_idx["val_mask"],args.lr,float(args.wd),args.epochs)
    test(model, graph, raw_features, label, split_idx["test_mask"])
