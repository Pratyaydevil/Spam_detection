import torch
import torch.nn as nn
import torch.nn.functional as F

class MetadataPipeline(nn.Module):
    def __init__(self, num_features, hidden_sizes=[128, 256, 128], output_size=128):
        super(MetadataPipeline, self).__init__()
        
        # We use a Tabular MLP to process structured data (URL lengths, header flags, etc)
        layers = []
        input_dim = num_features
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(input_dim, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, num_features)
        out = self.mlp(x)
        out = self.fc_out(out)
        return out
