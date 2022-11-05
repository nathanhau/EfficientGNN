import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import conv
import numpy as np


class DeepSGC(nn.module):
    def __init__(self,in_feats,n_hidden,n_classes,k,n_layers,activation,norm,dropout,iso,rewiring):
        super(DeepSGC,self).__init__()
        self.layers=nn.ModuleList()
        self.layer_k=k//n_layers
        if n_layers>1:
            self.layers.append(conv.sgconv(in_feats,n_hidden,layer_k+k%n_layers,True,norm=norm))
            for i in range(n_layers-2):
                self.layers.append(conv.sgconv(in_feats,n_hidden,layer_k,True,norm=norm))
            self.layers.append(conv.sgconv(in_feats,n_classes,layer_k,True,norm=norm))
        else:
            self.layers.append(conv.sgconv(in_feats,n_hidden,layer_k,True,norm=norm))
        
    
