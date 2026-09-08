import torch
import torch.nn as nn

from .text_pipeline import TextPipeline
from .metadata_pipeline import MetadataPipeline
from .fusion import AGCAF
from .evidential import DirichletEvidenceHead
from .novelty_extension import RetrievalAugmentedThreatIntel

class HybridEmailDefender(nn.Module):
    def __init__(self, num_meta_features, num_classes=3):
        super(HybridEmailDefender, self).__init__()
        
        # Branch 1: Semantic Text
        self.text_pipeline = TextPipeline()
        
        # Branch 2: Structured Metadata
        self.meta_pipeline = MetadataPipeline(num_features=num_meta_features)
        
        # Branch 3: Novelty 4 - RATI
        self.rati = RetrievalAugmentedThreatIntel()
        
        # Fusion Layer: Novelty 1 - AGCAF
        self.fusion = AGCAF()
        
        # Evidential Output: Novelty 3 - Epistemic Uncertainty
        # The fused dimension from AGCAF is 256
        self.evidence_head = DirichletEvidenceHead(input_dim=256, num_classes=num_classes)
        
    def forward(self, input_ids, attention_mask, meta_features):
        # 1. Process Text
        text_feats = self.text_pipeline(input_ids, attention_mask)
        
        # 2. Process Metadata
        meta_feats = self.meta_pipeline(meta_features)
        
        # 3. Retrieve Threat Intel
        rati_feats, rati_sims = self.rati(text_feats)
        
        # 4. Dynamic Fusion
        fused_feats, gates = self.fusion(text_feats, meta_feats, rati_feats)
        
        # 5. Evidential Classification
        prob, uncertainty, evidence, alpha = self.evidence_head(fused_feats)
        
        return {
            "probabilities": prob,
            "uncertainty": uncertainty,
            "evidence": evidence,
            "alpha": alpha,
            "gates": gates,
            "rati_similarity": rati_sims
        }
