import torch
import torch.nn as nn
import numpy as np

# In a full deployment, this would connect to FAISS/ChromaDB
# For the prototype, we simulate the retrieval mechanism

class RetrievalAugmentedThreatIntel(nn.Module):
    """
    Novelty 4: Retrieval-Augmented Threat Intelligence (RATI)
    Queries a vector database of known zero-day patterns using the text embedding.
    Returns a similarity score vector representing proximity to known threats.
    """
    def __init__(self, text_dim=256, output_dim=64):
        super(RetrievalAugmentedThreatIntel, self).__init__()
        
        # Maps the similarity score and retrieved metadata to a feature vector
        self.fc = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Simulating a small database of threat vectors (e.g. 5 known threat clusters)
        self.register_buffer('threat_db', torch.randn(5, text_dim))
        
    def forward(self, text_feats):
        # text_feats shape: (batch_size, text_dim)
        # Calculate cosine similarity with threat DB
        
        # Normalize text_feats
        t_norm = text_feats / (text_feats.norm(dim=1, keepdim=True) + 1e-8)
        
        # Normalize threat_db
        db_norm = self.threat_db / (self.threat_db.norm(dim=1, keepdim=True) + 1e-8)
        
        # Sim matrix: (batch_size, 5)
        similarities = torch.mm(t_norm, db_norm.transpose(0, 1))
        
        # Take the maximum similarity score for each item in the batch
        # This tells us how close it is to the nearest known zero-day threat
        max_sims, _ = torch.max(similarities, dim=1, keepdim=True)
        
        # We can also project the text feature directly combined with its max sim
        # Here we just pass the text feature through a projection layer modulated by similarity
        rati_feats = self.fc(text_feats) * max_sims
        
        return rati_feats, max_sims
