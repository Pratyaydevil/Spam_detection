import torch
import torch.nn as nn
import torch.nn.functional as F

class DirichletEvidenceHead(nn.Module):
    """
    Evidential Epistemic Uncertainty Estimation (Novelty 3).
    Outputs Dirichlet evidence parameters rather than standard Softmax probabilities.
    Allows explicit calculation of epistemic uncertainty for zero-day attacks.
    """
    def __init__(self, input_dim=256, num_classes=3):
        super(DirichletEvidenceHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Raw logits from the dense layer
        logits = self.fc(x)
        
        # Evidence is calculated using a softplus activation to ensure non-negativity
        # evidence = e_k >= 0
        evidence = F.softplus(logits)
        
        # Dirichlet parameters alpha_k = e_k + 1
        alpha = evidence + 1.0
        
        # Total evidence S = sum(alpha_k)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Probabilities p_k = alpha_k / S
        prob = alpha / S
        
        # Epistemic Uncertainty u = K / S (where K is num_classes)
        num_classes = alpha.shape[1]
        uncertainty = num_classes / S
        
        return prob, uncertainty, evidence, alpha
