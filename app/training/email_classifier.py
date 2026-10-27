from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from ..api.schemas import EmailRequest, XAIExplanation


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "email_classifier.joblib"
CLASS_NAMES = ("Ham", "Spam", "Phishing")


def build_classifier() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=60000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("character", TfidfVectorizer(
            analyzer="char",
            lowercase=True,
            ngram_range=(3, 5),
            max_features=40000,
            sublinear_tf=True,
            min_df=2,
        )),
    ])
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def load_classifier(path: Path = MODEL_PATH) -> Optional[Pipeline]:
    if not path.is_file():
        return None
    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline) or "features" not in loaded.named_steps or "classifier" not in loaded.named_steps:
        raise ValueError(f"Unsupported email classifier artifact: {path}")
    if not hasattr(loaded.named_steps["classifier"], "predict_proba"):
        raise ValueError(f"Email classifier artifact has no probability support: {path}")
    return loaded


def analyze_email(classifier: Pipeline, request: EmailRequest) -> Dict[str, Any]:
    probabilities_array = classifier.predict_proba([request.text])[0]
    estimator = classifier.named_steps["classifier"]
    classes = list(estimator.classes_)
    probabilities = {
        label: float(probabilities_array[classes.index(label)]) if label in classes else 0.0
        for label in CLASS_NAMES
    }
    classification = max(probabilities, key=probabilities.get)
    confidence = probabilities[classification]
    uncertainty = 1.0 - confidence

    explanations = _explain_prediction(classifier, request.text, classification)
    return {
        "classification": classification,
        "probabilities": probabilities,
        "epistemic_uncertainty": uncertainty,
        "requires_human_review": uncertainty > 0.6,
        "gating_weights": {"Text": 1.0, "Metadata": 0.0, "RATI (Threat Intel)": 0.0},
        "xai_breakdown": explanations,
        "baseline_comparison": {
            "baseline_model": "TF-IDF word/character n-grams + SGD",
            "baseline_prediction": classification,
            "baseline_confidence": confidence,
            "baseline_vulnerability": "Text-only model; sender authentication and URL metadata are not used.",
            "proposed_advantage": "A locally trained email-text classifier using labeled Ham, Spam, and Phishing examples.",
        },
        "analysis_mode": "trained_tfidf",
    }


def _explain_prediction(classifier: Pipeline, text: str, classification: str) -> List[XAIExplanation]:
    features = classifier.named_steps["features"]
    estimator = classifier.named_steps["classifier"]
    class_index = list(estimator.classes_).index(classification)
    transformed = features.transform([text])
    contributions = transformed.multiply(estimator.coef_[class_index]).toarray()[0]
    feature_names = features.get_feature_names_out()
    ranked_indices = np.argsort(contributions)[-3:][::-1]
    matched_terms = [
        feature_names[index].split("__", 1)[-1]
        for index in ranked_indices
        if contributions[index] > 0
    ]
    details = (
        "Text patterns that contributed to this class: " + ", ".join(matched_terms) + "."
        if matched_terms
        else "No individual text feature strongly influenced this prediction."
    )
    return [
        XAIExplanation(
            feature="Text n-gram signals",
            impact=1.0,
            description=details,
        )
    ]
