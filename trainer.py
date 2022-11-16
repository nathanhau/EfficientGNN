from models import sgc_precompute, ssgc_precompute, dgc_precompute, getNormAugAdj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
import dgl


def preprocess_linear(graph, features, model, K, T=None, alpha=None):
    precomputed = None
    pt = 0
    if (model == "sgc"):
        precomputed, pt = sgc_precompute(features, graph.adj(), K)
    elif (model == "ssgc"):
        assert isinstance(alpha, int), "Invalid alpha"
        precomputed, pt = ssgc_precompute(features, graph.adj(), K, alpha)

    elif (model == "dgc"):
        assert isinstance(T, int) or isinstance(T, float), "Invalid T"
        precomputed, pt = dgc_precompute(features, graph.adj(), T, K)
    else:
        raise ValueError("Invalid model")
    return precomputed, pt


def train_linear(features, labels, n_classes, epochs=100, lr=0.2, weight_decay=5e-6):
    feature_size = features.shape[1]
    labels = F.one_hot(labels, num_classes=n_classes).squeeze().to(torch.float)
    ln = nn.Linear(feature_size, n_classes)
    optimizer = optim.Adam(ln.parameters(), lr=lr, weight_decay=weight_decay)
    t = perf_counter()
    for i in range(epochs):
        ln.train()
        optimizer.zero_grad()
        output = ln(features)
        print(output.shape)
        print(labels.shape)
        loss = F.cross_entropy(output, labels)
        print(loss)
        loss.backward()
        optimizer.step()
    training_time = perf_counter()-t
    return ln, training_time


def test(model, g, features, labels, train_mask, val_mask, test_mask, loss_fn=nn.CrossEntropyLoss(), device='cpu'):
    model.eval()
    logits = model(g, features)
    train_acc = torch.sum(logits[train_mask].argmax(1) == labels[train_mask]).item() / train_mask.sum().item()
    val_acc = torch.sum(logits[val_mask].argmax(1) == labels[val_mask]).item() / val_mask.sum().item()
    test_acc = torch.sum(logits[test_mask].argmax(1) == labels[test_mask]).item() / test_mask.sum().item()
    return train_acc, val_acc, test_acc


def train(model, g, features, labels, train_mask, val_mask, test_mask, epochs=100, loss_fn=nn.CrossEntropyLoss(), optimizer=optim.Adam, lr=0.01, weight_decay=5e-4, device='cpu'):
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, g, features, labels, train_mask, val_mask, test_mask, loss_fn, device)
            print(f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')