import torch
import torch.nn as nn
import torch.nn.functional as F

class AvocadoNet(nn.Module):
    def __init__(self, in_params=8, hidden_dims=10):
        super(AvocadoNet, self).__init__()
        self.fc1 = nn.Linear(in_params, hidden_dims)
        # activation
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)
        self.out_shape = (1,)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
