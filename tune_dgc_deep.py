import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import load_data, preprocess_graph
from models.SGCRes import SGCRes
from models.SSGC import SSGC
from models.DGC import DGC
from models.SGC import SGC
from stacked_models import DeepLinear, LinearMLP
from models.GCN import GCN
from dgl.nn.pytorch.conv import SGConv
from time import perf_counter





class Graph:
    graph, labels, split_idx = load_data('cora')
    n_layers=[]

def objective(trial):
    
    graph, labels, split_idx = Graph.graph, Graph.labels, Graph.split_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = preprocess_graph(graph).to(device)
    labels=labels.to(device)
    # print(graph.ndata['feat'])
    features = graph.ndata['feat'].to(device)
    is_linear=False
    in_feats = features.shape[-1]
    n_classes = labels.max().item()+1
    K=trial.suggest_int("K",4,30,2)
    # K=trial.suggest_int("K",2,10,log=True)
    # K=sum(Graph.n_layers)

    T=trial.suggest_float("T",1,10)
    loss_fn=nn.CrossEntropyLoss()
    n_hidden=32
    model=DeepLinear("DGC", {"T":T}, in_feats, n_hidden, n_classes, K, Graph.n_layers, nn.ReLU(), device)
    epochs=100
    train_mask,val_mask=split_idx["train_mask"], split_idx["val_mask"]
    lr=trial.suggest_float("lr",0.005,0.1,log=True)
    # lr=0.2
    weight_decay=trial.suggest_float('weight_decay',1e-4,1e-3)
    
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
        loss.backward()
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
        trial.report(val_acc,epoch)
    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')
    return val_acc

if __name__ == "__main__":
    f=open("deepdgc_tune.txt","a")
    
    comb=[2,4]
    for i in range(len(comb)):
        study = optuna.create_study(direction="maximize")
        Graph.n_layers=comb[i]
        study.optimize(objective, n_trials=400,timeout=3600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        f.write(str(comb[i])+"\n")
        a=f"    Value: {trial.value}"
        f.write(a+'\n')
        print(a)
        f.write("  Params: \n")
        print("  Params: ")
        for key, value in trial.params.items():
            b="    {}: {}".format(key, value)
            print(b)
            f.write(b+'\n')
    f.close()