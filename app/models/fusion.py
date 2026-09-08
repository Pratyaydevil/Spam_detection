import torch
import torch.nn as nn
import torch.nn.functional as F

class AGCAF(nn.Module):
    """
    Adaptive Gated Cross-Attention Fusion (AGCAF)
    Dynamically down-weights the metadata branch if it appears manipulated or spoofed,
    relying more on the semantic text features.
    """
    def __init__(self, text_dim=256, meta_dim=128, rati_dim=64, fused_dim=256):
        super(AGCAF, self).__init__()
        
        # Projections to common space for attention
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.meta_proj = nn.Linear(meta_dim, fused_dim)
        self.rati_proj = nn.Linear(rati_dim, fused_dim)
        
        # Attention gate to calculate the trust factor $g$
        # It takes the concatenated text and meta representations to determine trust in metadata
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.Tanh(),
            nn.Linear(fused_dim, 3), # outputs 3 weights for text, meta, rati
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Linear(fused_dim * 3, fused_dim)
        
    def forward(self, text_feats, meta_feats, rati_feats):
        # 1. Project to common dimension
        t_proj = self.text_proj(text_feats)
        m_proj = self.meta_proj(meta_feats)
        r_proj = self.rati_proj(rati_feats)
        
        # 2. Calculate dynamic gating weights
        concat_feats = torch.cat([t_proj, m_proj, r_proj], dim=-1)
        gates = self.gate(concat_feats) # shape: (batch_size, 3)
        
        g_text = gates[:, 0].unsqueeze(1)
        g_meta = gates[:, 1].unsqueeze(1)
        g_rati = gates[:, 2].unsqueeze(1)
        
        # 3. Apply gates
        weighted_t = t_proj * g_text
        weighted_m = m_proj * g_meta
        weighted_r = r_proj * g_rati
        
        # 4. Fusion
        fused = torch.cat([weighted_t, weighted_m, weighted_r], dim=-1)
        fused_out = F.relu(self.output_proj(fused))
        
        return fused_out, gates
