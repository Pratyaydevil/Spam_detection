from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import DistilBertTokenizer
from pydantic import ValidationError

from .schemas import EmailRequest, AnalysisResponse, XAIExplanation
from ..models.hybrid_model import HybridEmailDefender

app = FastAPI(
    title="Hybrid ML-DNN Spam & Phishing Defender",
    description="API for adaptive, robust, and explainable email classification with RATI novelty.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models globally (in a real app, this would load pre-trained weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_meta_features = 6 # defined in EmailRequest
model = HybridEmailDefender(num_meta_features=num_meta_features, num_classes=3).to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
classes = ["Ham", "Spam", "Phishing"]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_email(request: EmailRequest):
    try:
        # 1. Tokenize Text
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 2. Extract Metadata
        meta_features = torch.tensor([[
            request.url_length,
            float(request.subdomain_count),
            float(request.https_presence),
            float(request.spf_dkim_failure),
            request.sender_reputation,
            float(request.recipient_count)
        ]], dtype=torch.float32).to(device)
        
        # 3. Model Inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, meta_features)
            
        prob = outputs["probabilities"][0].cpu().numpy()
        uncertainty = outputs["uncertainty"][0].cpu().item()
        gates = outputs["gates"][0].cpu().numpy()
        
        predicted_idx = prob.argmax()
        classification = classes[predicted_idx]
        
        # Simulate XAI Output for demonstration
        # In a real setup, we would use Captum (Integrated Gradients) here
        xai_mock = [
            XAIExplanation(feature="SPF/DKIM Failure", impact=0.45 if request.spf_dkim_failure else 0.05, description="Authentication records failed or missing."),
            XAIExplanation(feature="Semantic Urgency", impact=0.35, description="Language indicates high urgency or threat.")
        ]
        
        threshold = 0.6  # High epistemic uncertainty threshold for zero-day flag
        
        return AnalysisResponse(
            classification=classification,
            probabilities={classes[i]: float(prob[i]) for i in range(3)},
            epistemic_uncertainty=uncertainty,
            requires_human_review=(uncertainty > threshold),
            gating_weights={
                "Text": float(gates[0]),
                "Metadata": float(gates[1]),
                "RATI (Threat Intel)": float(gates[2])
            },
            xai_breakdown=xai_mock
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
