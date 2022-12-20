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







def objective(trial):
    
    graph, labels, split_idx = load_data('cora')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = preprocess_graph(graph).to(device)
    labels=labels.to(device)
    # print(graph.ndata['feat'])
    features = graph.ndata['feat'].to(device)

    in_feats = features.shape[-1]
    n_classes = labels.max().item()+1
    #K=2
    K=trial.suggest_int('K',2,20,log=True)
    is_linear=False
    loss_fn=nn.CrossEntropyLoss()
    model=SGC(in_feats,n_classes,K,device,is_linear=True)
    epochs=100
    epochs=trial.suggest_int('epochs',50,100)
    train_mask,val_mask=split_idx["train_mask"], split_idx["val_mask"]
    #lr=0.2
    lr=trial.suggest_float("lr",1e-5,1e-3)
    #weight_decay=1.2e-05
    weight_decay=trial.suggest_float('weight_decay',1e-6,1e-3,log=True)
    
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
        # print(f'Epoch {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
        trial.report(val_acc,epoch)
    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')
    return val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_contour(study)