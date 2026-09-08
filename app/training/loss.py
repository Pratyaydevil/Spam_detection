import torch
import torch.nn as nn
import torch.nn.functional as F

class SCARLoss(nn.Module):
    """
    Novelty 2: Supervised Contrastive Adversarial Regularization (SCAR)
    Combines Evidential Loss with Supervised Contrastive Loss to map obfuscated
    text vectors directly onto their unperturbed, original anchor embeddings.
    """
    def __init__(self, num_classes=3, lambda_contrastive=0.1):
        super(SCARLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_contrastive = lambda_contrastive
        
    def evidential_loss(self, alpha, y):
        # y is one-hot encoded true labels
        S = torch.sum(alpha, dim=1, keepdim=True)
        # Expected Probability: E[p_k] = alpha_k / S
        # CE loss is sum(y_k * (log(S) - log(alpha_k)))
        loss_ce = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1)
        
        # KL Divergence regularization to shrink evidence for incorrect classes
        # This penalizes models that output high evidence for everything
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1) \
                 - torch.lgamma(torch.ones_like(S_tilde) * self.num_classes)
        
        kl_div += torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1)
        
        # Combine base CE and KL divergence
        # we weight KL by epoch or step in a real scenario, here fixed small weight
        loss = loss_ce + 0.1 * kl_div
        return torch.mean(loss)
        
    def contrastive_loss(self, anchor_feats, adv_feats):
        # Enforce that the adversarial feature representation is close to the anchor
        # Mean Squared Error or Cosine Embedding Loss
        loss = F.mse_loss(adv_feats, anchor_feats)
        return loss
        
    def forward(self, alpha, y, anchor_feats=None, adv_feats=None):
        # Standard Evidential Loss
        loss_ev = self.evidential_loss(alpha, y)
        
        # SCAR Component
        loss_sc = 0.0
        if anchor_feats is not None and adv_feats is not None:
            loss_sc = self.contrastive_loss(anchor_feats, adv_feats)
            
        total_loss = loss_ev + self.lambda_contrastive * loss_sc
        
        return total_loss, loss_ev, loss_sc
