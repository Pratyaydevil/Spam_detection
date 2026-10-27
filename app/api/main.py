from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .schemas import EmailRequest, AnalysisResponse
from ..training.email_classifier import MODEL_PATH, analyze_email as classify_email_text, load_classifier

app = FastAPI(
    title="Local Email Spam and Phishing Classifier",
    description="API for a locally trained TF-IDF text classifier.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


classifier = load_classifier()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_email(request: EmailRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No trained email classifier is installed. Run python -m app.training.train_email_classifier first.",
        )
    return classify_email_text(classifier, request)

@app.get("/benchmark")
async def get_benchmark_metrics():
    """Returns illustrative chart data that is not measured from the current text classifier."""
    return {
        "note": "Illustrative chart data only; see the generated training_metrics.json for held-out results from the current classifier.",
        "kpis": {
            "accuracy": 99.4,
            "f1_score": 99.3,
            "adversarial_robustness": 98.6,
            "zero_day_detection": 93.8,
            "latency_ms": 24.6,
            "false_positive_rate": 0.35
        },
        "models": [
            {
                "name": "Naive Bayes (TF-IDF)",
                "category": "Classical ML",
                "accuracy": 86.4,
                "precision": 84.8,
                "recall": 87.1,
                "f1": 85.9,
                "roc_auc": 90.2,
                "robustness": 41.3,
                "zero_day": 22.0,
                "latency": 2.1,
                "is_ours": False
            },
            {
                "name": "SVM (Linear + Heuristics)",
                "category": "Classical ML",
                "accuracy": 90.1,
                "precision": 89.3,
                "recall": 90.5,
                "f1": 89.9,
                "roc_auc": 93.4,
                "robustness": 48.5,
                "zero_day": 28.4,
                "latency": 5.4,
                "is_ours": False
            },
            {
                "name": "Random Forest (Metadata + TF-IDF)",
                "category": "Classical ML",
                "accuracy": 92.5,
                "precision": 91.8,
                "recall": 92.9,
                "f1": 92.3,
                "roc_auc": 95.1,
                "robustness": 54.2,
                "zero_day": 34.1,
                "latency": 8.7,
                "is_ours": False
            },
            {
                "name": "XGBoost (URL + Header Features)",
                "category": "Classical ML",
                "accuracy": 94.2,
                "precision": 93.7,
                "recall": 94.5,
                "f1": 94.1,
                "roc_auc": 96.8,
                "robustness": 59.0,
                "zero_day": 39.6,
                "latency": 11.2,
                "is_ours": False
            },
            {
                "name": "BiLSTM (Text Only)",
                "category": "Deep Learning",
                "accuracy": 93.8,
                "precision": 93.2,
                "recall": 94.0,
                "f1": 93.6,
                "roc_auc": 96.2,
                "robustness": 61.4,
                "zero_day": 48.2,
                "latency": 18.5,
                "is_ours": False
            },
            {
                "name": "DistilBERT (Text Transformer)",
                "category": "Deep Learning",
                "accuracy": 95.6,
                "precision": 95.1,
                "recall": 95.8,
                "f1": 95.4,
                "roc_auc": 97.7,
                "robustness": 64.2,
                "zero_day": 52.7,
                "latency": 21.3,
                "is_ours": False
            },
            {
                "name": "Concat Fusion (DistilBERT + Tabular MLP)",
                "category": "Hybrid Ensemble",
                "accuracy": 96.7,
                "precision": 96.3,
                "recall": 96.9,
                "f1": 96.6,
                "roc_auc": 98.3,
                "robustness": 68.4,
                "zero_day": 58.3,
                "latency": 23.1,
                "is_ours": False
            },
            {
                "name": "Hybrid ML-DNN (Ours - AGCAF + SCAR + Evidential)",
                "category": "Proposed Framework ⭐",
                "accuracy": 99.4,
                "precision": 99.2,
                "recall": 99.5,
                "f1": 99.3,
                "roc_auc": 99.9,
                "robustness": 98.6,
                "zero_day": 93.8,
                "latency": 24.6,
                "is_ours": True
            }
        ],
        "per_class": [
            {"class_name": "Ham (Legitimate)", "precision": 99.8, "recall": 99.7, "f1": 99.7, "support": 4200},
            {"class_name": "Spam (Commercial / Bulk)", "precision": 99.4, "recall": 99.5, "f1": 99.4, "support": 2850},
            {"class_name": "Phishing (Credential Spoofing)", "precision": 99.1, "recall": 99.2, "f1": 99.1, "support": 1950}
        ],
        "ablation": [
            {"step": "DistilBERT Text Baseline", "f1": 95.4, "robustness": 64.2, "zero_day": 52.7, "delta": "Baseline", "notes": "Contextual semantic baseline without metadata integration."},
            {"step": "+ Tabular MLP (Static Concat)", "f1": 96.6, "robustness": 68.4, "zero_day": 58.3, "delta": "+1.2% F1", "notes": "Blind feature concatenation; vulnerable to spoofed metadata."},
            {"step": "+ AGCAF Dynamic Gating", "f1": 97.9, "robustness": 79.5, "zero_day": 67.4, "delta": "+1.3% F1, +11.1% Rob.", "notes": "Dynamically downweights metadata when header spoofing is detected."},
            {"step": "+ SCAR Contrastive Loss", "f1": 98.7, "robustness": 98.6, "zero_day": 74.2, "delta": "+0.8% F1, +19.1% Rob.", "notes": "Topological anchor forces obfuscated text to map to canonical representations."},
            {"step": "+ Evidential Head & RATI (Full Proposed)", "f1": 99.3, "robustness": 98.6, "zero_day": 93.8, "delta": "+0.6% F1, +19.6% Zero-Day", "notes": "Dirichlet prior + vector threat retrieval flags zero-day attacks with epistemic certainty."}
        ],
        "confusion_matrix": {
            "classes": ["Ham", "Spam", "Phishing"],
            "matrix": [
                [4188, 8, 4],
                [7, 2836, 7],
                [5, 11, 1934]
            ],
            "total_samples": 9000,
            "overall_accuracy": 99.53,
            "overall_f1": 99.42
        },
        "roc_curves": [
            {
                "id": "curve_ours",
                "name": "★ Hybrid ML-DNN (Ours)",
                "auc": 0.999,
                "color": "#c084fc",
                "stroke_width": 3.5,
                "is_ours": True,
                "visible": True,
                "points": [[0,0],[0.001,0.92],[0.003,0.97],[0.006,0.992],[0.012,0.997],[0.03,0.999],[0.1,1.0],[1.0,1.0]]
            },
            {
                "id": "curve_concat",
                "name": "Concat Fusion (Base Hybrid)",
                "auc": 0.983,
                "color": "#60a5fa",
                "stroke_width": 2.4,
                "is_ours": False,
                "visible": True,
                "points": [[0,0],[0.01,0.84],[0.025,0.91],[0.05,0.95],[0.1,0.98],[0.2,0.99],[1.0,1.0]]
            },
            {
                "id": "curve_distilbert",
                "name": "DistilBERT (Text Transformer)",
                "auc": 0.977,
                "color": "#22d3ee",
                "stroke_width": 2.2,
                "is_ours": False,
                "visible": True,
                "points": [[0,0],[0.015,0.80],[0.035,0.88],[0.07,0.94],[0.15,0.97],[0.3,0.99],[1.0,1.0]]
            },
            {
                "id": "curve_xgboost",
                "name": "XGBoost (Tabular URL/Headers)",
                "auc": 0.968,
                "color": "#fbbf24",
                "stroke_width": 2.0,
                "is_ours": False,
                "visible": True,
                "points": [[0,0],[0.02,0.76],[0.05,0.86],[0.1,0.93],[0.2,0.96],[0.4,0.98],[1.0,1.0]]
            },
            {
                "id": "curve_rf",
                "name": "Random Forest",
                "auc": 0.951,
                "color": "#f472b6",
                "stroke_width": 1.8,
                "is_ours": False,
                "visible": True,
                "points": [[0,0],[0.03,0.72],[0.07,0.82],[0.14,0.91],[0.25,0.95],[0.5,0.98],[1.0,1.0]]
            },
            {
                "id": "curve_nb",
                "name": "Naive Bayes (Baseline)",
                "auc": 0.902,
                "color": "#94a3b8",
                "stroke_width": 1.6,
                "is_ours": False,
                "visible": True,
                "points": [[0,0],[0.05,0.62],[0.12,0.76],[0.22,0.85],[0.4,0.91],[0.7,0.96],[1.0,1.0]]
            }
        ],
        "radar_metrics": {
            "axes": [
                "Clean Accuracy",
                "Adversarial Robustness",
                "Zero-Day Detection",
                "Spoof Invariance",
                "Latency Efficiency",
                "FPR Control"
            ],
            "series": [
                {
                    "name": "Hybrid ML-DNN (Ours)",
                    "color": "#a855f7",
                    "values": [99.4, 98.6, 93.8, 97.9, 88.0, 99.6],
                    "is_ours": True
                },
                {
                    "name": "DistilBERT Baseline",
                    "color": "#06b6d4",
                    "values": [95.6, 64.2, 52.7, 68.4, 89.0, 96.2],
                    "is_ours": False
                },
                {
                    "name": "XGBoost Baseline",
                    "color": "#f59e0b",
                    "values": [94.2, 59.0, 39.6, 84.1, 95.0, 95.8],
                    "is_ours": False
                }
            ]
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model_path": str(MODEL_PATH),
        "analysis_mode": "trained_tfidf" if classifier is not None else "unavailable",
    }

# --- Serve Frontend ---
# Resolve the frontend directory relative to this file's location
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Mount static files (CSS, JS) AFTER API routes so /analyze etc. take priority
app.mount("/", StaticFiles(directory=frontend_dir), name="static")
