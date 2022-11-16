import torch.nn as nn

device = 'cpu'
activation = nn.ReLU()
epochs = 100
batch_size = 10000
lr = 0.02
loss_fn = nn.CrossEntropyLoss()
weight_decay = 5e-4
n_layers = 2
dropout = 0.2
