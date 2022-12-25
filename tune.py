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
from functools import partial
from modularized_training import train, test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Run:
    def __init__(self,name,config):
        self.name=name
        self.graph, self.labels, self.split_idx = load_data(name)
        self.config=config if isinstance(config,list) else [config]
        self.graph = preprocess_graph(self.graph).to(device)
        self.labels=self.labels.to(device)
        self.features = self.graph.ndata['feat'].to(device)
        self.in_feats = self.features.shape[-1]
        self.n_classes = self.labels.max().item()+1
        

def getParam(trial,p,name):
    param=p.copy()
    # param['tune']=False
    K=param.get('K',None)
    if isinstance(K,list):
        param['K']=trial.suggest_int('K',K[0],K[1],2)
        # param['tune']=True
    else:
        param['K']=K
    h=param.get('n_hidden',None)
    if h is None:
        param['n_hidden']=32
    default_lim=[[0.01,0.1,True],[5e-6,1e-3,False],[0.1,min(10,param['K']),False],[0,1,False]]
    for i,j in enumerate(['lr','wd','T','alpha']):
        a=param.get(j,None)
        if a is None:
            if (j=='lr'or j=='wd' or (j=='T' and (name=='dgc' or name=='deepdgc') 
                or (j=='alpha' and (name=='ssgc' or name=='deepssgc' or name=='deepsgcres')))):
                param[j]=trial.suggest_float(j,default_lim[i][0],default_lim[i][1],log=default_lim[i][2])
                param['tune']=True
        elif isinstance(a,list):
            param[j]=trial.suggest_float(j,a[0],a[1])
            param['tune']=True
        else:
            param[j]=a
        # print(str(param))
    return param
    

def objective(run,name,p,trial):
    
    graph, labels, split_idx = run.graph, run.labels, run.split_idx
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels=run.labels
    # print(run.ndata['feat'])
    # n_hidden=64
    features = run.features
    is_linear=False
    in_feats = run.in_feats
    n_classes = run.n_classes
    param=getParam(trial, p,name)
    # K=trial.suggest_int("K",4,30,2)
    # K=trial.suggest_int("K",2,10,log=True)
    # K=sum(run.n_layers)
    # K=run.p[run.id]['K'] if run.p[run.id]['K'] is not None else trial.suggest_int("K",2,10,2)

    # a=trial.suggest_float("alpha",0.01,1)
    loss_fn=nn.CrossEntropyLoss()
    print(str(param))
    model=getModel(name, param, in_feats, n_classes)
    model=model.to(device)
    epochs=100
    train_mask,val_mask=split_idx["train_mask"], split_idx["val_mask"]
    # lr=trial.suggest_float("lr",0.01,0.2)
    # lr=0.01
    # wd=trial.suggest_float('weight_decay',1e-7,1e-5)
    
    t = perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['wd'])
    train_mask=train_mask.to(device)
    val_mask=val_mask.to(device)
    labels=labels.to(device)
    graph=graph.to(device)
    features=features.to(device)
    for epoch in range(epochs):
        model.train()
        logits = None
        if is_linear:
            logits = model(features)
        else:
            logits = model(graph, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if is_linear:
            logits = model(features)
        else:
            logits = model(graph, features)
        train_acc = torch.sum(logits[train_mask].argmax(1) == labels[train_mask]).item() / train_mask.sum().item()
        val_acc = torch.sum(logits[val_mask].argmax(1) == labels[val_mask]).item() / val_mask.sum().item()
        # test_acc = torch.sum(logits[test_mask].argmax(1) == labels[test_mask]).item() / test_mask.sum().item()
        if (epoch+1)%10==0:
            print(f'Epoch {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
        trial.report(val_acc,epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    training_time = perf_counter()-t
    print(f'Training time: {training_time:.4f}s')
    return val_acc

def getModel(name,param,in_feats,n_classes):
    if name=='sgc':
        return SGC(in_feats, n_classes, param['K'], device,is_linear=True)
    elif name=='dgc':
        return DGC(in_feats, n_classes, param['K'], param['T'], device,is_linear=True)
    elif name=='ssgc':
        return SSGC(in_feats, n_classes, param['K'], param['alpha'], device,is_linear=True)
    elif name=='gcn':
        return GCN(in_feats, param['n_hidden'], n_classes, param['n_layers'], nn.ReLU(), device)
    elif name.find('deep')!=-1:
        return DeepLinear(name[name.find('deep')+4:], param, in_feats, param['n_hidden'], n_classes, param['K'], param['n_layers'], nn.ReLU(), device)
    return model

def runTrial(run,model_name,config):
    fname=run.name+'_'+model_name+'_tune.txt'
    
    
    for p in config:
        f=open(fname,"a")
        # run.id=i
        study = optuna.create_study(direction="maximize",pruner=optuna.pruners.HyperbandPruner(
        min_resource=5, max_resource=100, reduction_factor=3
        ))
        
        # run.n_layers=comb[i]
        obj=partial(objective,run,model_name,p)
        study.optimize(obj,n_trials=250,timeout=1200)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print(str(p))
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        f.write(str(p)+"\n")
        f.write("val: "+str(trial.value)+ " "+str(len(study.trials))+": "
                +str(len(pruned_trials))+"/"+str(len(complete_trials))+'\n')
        a=f"    Value: {trial.value}"
        # f.write(a+'\n')
        print(a)
        f.write("  Params: \n")
        print("  Params: ")
        for key, value in trial.params.items():
            b="    {}: {}".format(key, value)
            print(b)
            f.write(b+'\n')
            p[key]=value
        
        result=[]
        for i in range(5):
            best=getModel(model_name, p, run.in_feats, run.n_classes)
            train(best, run.graph, run.features, run.labels, run.split_idx["train_mask"], 
                run.split_idx["val_mask"], p['lr'], p['wd'], 100)
            result.append(test(best,run.graph,run.features,run.labels,run.split_idx["test_mask"]))
        testmsg="Test result: "+" ".join([str(i) for i in result])+" average: "+str(sum(result)/len(result))
        print(testmsg)
        f.write(testmsg+'\n')
        f.close()

def runTest(run,model_name,config):
    fname=run.name+'_'+model_name+'_tune.txt'
    
    
    for p in config:
        f=open(fname,"a")
        # run.id=i
        f.write(str(p)+"\n")
        result=[]
        for i in range(5):
            best=getModel(model_name, p, run.in_feats, run.n_classes)
            train(best, run.graph, run.features, run.labels, run.split_idx["train_mask"], 
                run.split_idx["val_mask"], p['lr'], p['wd'], 100)
            result.append(test(best,run.graph,run.features,run.labels,run.split_idx["test_mask"]))
        testmsg="Test result: "+" ".join([str(i) for i in result])+" average: "+str(sum(result)/len(result))
        print(testmsg)
        f.write(testmsg+'\n')
        f.close()
        
    
    # print(a)
if __name__=="__main__":
    # config=[
    #     {'name':'deepsgcres','config':[{'K':4, 'n_layers':2},{'K':6, 'n_layers':3},
    #         {'K':6, 'n_layers':2},{'K':8, 'n_layers':2},{'K':10, 'n_layers':2},{'K':50, 'n_layers':2},
    #         {'K':80, 'n_layers':2},{'K':6, 'n_layers':[4,2]},{'K':6, 'n_layers':[2,4]}]},
    #     {'name':'deepdgc','config':[{'K':4, 'n_layers':2},{'K':6, 'n_layers':3},
    #         {'K':6, 'n_layers':2},{'K':8, 'n_layers':2},{'K':10, 'n_layers':2},{'K':50, 'n_layers':2},
    #         {'K':80, 'n_layers':2},{'K':6, 'n_layers':[4,2]},{'K':6, 'n_layers':[2,4]}]},
    #     {'name':'deepsgc','config':[{'K':4, 'n_layers':2},{'K':6, 'n_layers':3},
    #         {'K':6, 'n_layers':2},{'K':8, 'n_layers':2},{'K':10, 'n_layers':2},{'K':50, 'n_layers':2},
    #         {'K':80, 'n_layers':2},{'K':6, 'n_layers':[4,2]},{'K':6, 'n_layers':[2,4]}]}
    # ]
    config=[{'name':'deepsgc', 'config':[{'K':2**i,'lr': 0.05, 'wd':5e-04, 'n_layers':2,'n_hidden':32} for i in range(7,9) ]}]
    for d in ['cora']:
        exp=Run(d,config)
        for m in exp.config:
            runTest(exp, m['name'], m['config'])
        del exp
        