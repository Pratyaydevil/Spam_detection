from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class EmailRequest(BaseModel):
    text: str = Field(..., description="The raw email body text")
    url_length: float = Field(..., description="Average length of URLs in the email")
    subdomain_count: int = Field(..., description="Maximum number of subdomains in any URL")
    https_presence: int = Field(..., description="1 if HTTPS is predominantly used, 0 otherwise")
    spf_dkim_failure: int = Field(..., description="1 if SPF/DKIM validation failed, 0 otherwise")
    sender_reputation: float = Field(..., description="Score from 0.0 (bad) to 1.0 (good)")
    recipient_count: int = Field(..., description="Number of recipients in To/Cc/Bcc")

class XAIExplanation(BaseModel):
    feature: str
    impact: float
    description: str

class AnalysisResponse(BaseModel):
    classification: str = Field(..., description="Ham, Spam, or Phishing")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    epistemic_uncertainty: float = Field(..., description="Uncertainty score (higher means model is guessing)")
    requires_human_review: bool = Field(..., description="True if epistemic_uncertainty > threshold")
    gating_weights: Dict[str, float] = Field(..., description="How much the model trusted Text vs Metadata vs RATI")
    xai_breakdown: List[XAIExplanation] = Field(..., description="Explanation of features driving the decision")
