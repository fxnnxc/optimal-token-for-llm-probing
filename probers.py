import torch 
import torch.nn as nn 

def get_prober(prober_type, model_dim, num_layers=1, init_seed=42, **kwargs):
    if prober_type == "BCEProber":
        return BCEProber(model_dim, num_layers, init_seed=init_seed, **kwargs)
    else:
        raise ValueError(f"Prober type {prober_type} not supported")


class BCEProber(nn.Module):
    def __init__(self, model_dim, num_layers=1, hidden_dim=256, bias=True, init_seed=42, **kwargs):
        super().__init__()
        if num_layers ==1:
            self.model = nn.Linear(model_dim, 1, bias=bias)
        elif num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1, bias=bias)
            )
        else:
            raise ValueError(f"Number of layers {num_layers} not supported")
        self.init_seed = init_seed
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(self.init_seed)
        if hasattr(self.model, 'weight') and self.model.weight is not None:
            nn.init.xavier_uniform_(self.model.weight)
        if hasattr(self.model, 'bias') and self.model.bias is not None:
            nn.init.zeros_(self.model.bias)
            
    def forward(self, x):
        if x.ndim == 3:
            x = x[:, -1, :]
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs  # or return logits, probs if you want both

    def compute_loss(self, logits, y):
        y = y.float().unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        return loss
    
    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            probs = self.forward(x).squeeze(1)
            preds = (probs > threshold).long()
        return preds, probs
