import torch
import torch.nn as nn
from dgl.nn.pytorch import conv


class DeepSGC(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, k, n_layers, activation, bias=True, norm=False, dropout=0, iso=False, adj=None, rewiring=False):
        super(DeepSGC, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.k = k
        self.activation = activation
        self.bias = bias
        self.norm = norm,
        self.dropout = nn.Dropout(p=dropout)
        self.iso = iso
        self.adj = adj
        self.layers = nn.ModuleList()
        self.klist = []
        if not isinstance(n_layers, list):
            self.klist = [k//n_layers for i in range(n_layers)]
            self.klist[0] += k % n_layers
            self.n_layers = n_layers
        else:
            self.klist = n_layers
            self.n_layers = len(n_layers)

        if self.n_layers > 1:
            self.layers.append(conv.SGConv(in_feats, n_hidden, self.klist[0], True, bias=bias, norm=norm))
            for i in range(1, n_layers-1):
                self.layers.append(conv.SGConv(in_feats, n_hidden, self.klist[i], True, bias=bias, norm=norm))
            self.layers.append(conv.SGConv(in_feats, n_classes, self.klist[-1], True, bias=bias, norm=norm))
        else:
            self.layers.append(conv.SGConv(in_feats, n_hidden, self.klist[0], True, bias=bias, norm=norm))
        if self.iso and (self.adj != None):
            stdv = self.get_std()
            for layer in self.layers:
                for i in range(0, layer.fc.weight.shape[0]):
                    layer.fc.weight[i].data.uniform_(-stdv, stdv)
                # nn.init.uniform_(layer.fc.weight,-stdv,stdv)
                if self.bias is True:
                    layer.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < self.layers.len()-1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def get_std(self):
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim=1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim=0).to_dense().unsqueeze(0) + 1
        # support = torch.sqrt(torch.sum(d_i * d_j))
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        # print(stdv)
        return stdv