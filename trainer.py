from models import sgc_precompute,ssgc_precompute,dgc_precompute, getNormAugAdj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter

def preprocess_linear(graph,features,model,K,T=None,alpha=None):
    precomputed=None
    pt=0
    if (model=="sgc"):
        precomputed,pt=sgc_precompute(features,graph.adj(),K)
    elif (model=="ssgc"):
        assert isinstance(alpha, int), "Invalid alpha"
        precomputed,pt=ssgc_precompute(features,graph.adj(),K,alpha)

    elif (model=="dgc"):
        assert isinstance(T, int) or isinstance(T, float), "Invalid T"
        precomputed,pt=dgc_precompute(features,graph.adj(),T,K)
    else:
        raise ValueError("Invalid model")
    return precomputed,pt

def train_linear(features,labels,n_classes,epochs=100,lr=0.2,weight_decay=5e-6):
    feature_size=features.shape[1]
    labels=F.one_hot(labels,num_classes=n_classes).squeeze().to(torch.float)
    ln=nn.Linear(feature_size,n_classes)
    optimizer=optim.Adam(ln.parameters(),lr=lr,weight_decay=weight_decay)
    t=perf_counter();
    for i in range(epochs):
        ln.train()
        optimizer.zero_grad()
        output=ln(features)
        print(output.shape)
        print(labels.shape)
        loss=F.cross_entropy(output,labels)
        print(loss)
        loss.backward()
        optimizer.step( )
    training_time=perf_counter()-t
    return ln, training_time

