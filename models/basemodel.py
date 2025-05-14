import torch.nn as nn

class BaseModel(nn.Module):
    """Base model for all models
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError