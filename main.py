import dgl
import torch.nn.functional as F
import numpy as np
from DeepSGC import DeepSGC

g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
g = dgl.add_self_loop(g)
model = DeepSGC(10, 6, 3, 7, 3, F.relu, iso=True, adj=g.adj())
for i in model.layers:
    print(i.fc.weight.shape)
    print(i._k)
