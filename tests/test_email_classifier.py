import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import EmailRequest
from app.training.email_classifier import analyze_email, build_classifier
from app.training.train_email_classifier import load_labeled_emails


class EmailClassifierTests(unittest.TestCase):
    def test_api_refuses_inference_without_a_trained_model(self):
        with patch("app.api.main.classifier", None):
            response = TestClient(app).post("/analyze", json={
                "text": "Test email text with enough content to analyze.",
                "url_length": 0,
                "subdomain_count": 0,
                "https_presence": 1,
                "spf_dkim_failure": 0,
                "sender_reputation": 0.5,
                "recipient_count": 1,
            })
        self.assertEqual(response.status_code, 503)
        self.assertIn("train_email_classifier", response.json()["detail"])

    def test_dataset_labels_map_to_three_classes_and_duplicates_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            enron_path = Path(directory) / "enron.csv"
            phishing_path = Path(directory) / "phishing.csv"
            pd.DataFrame({
                "email": [
                    "This is a long legitimate internal email about an engineering meeting.",
                    "Buy discounted medicine from this online store right now today.",
                ],
                "label": [0, 1],
            }).to_csv(enron_path, index=False)
            pd.DataFrame({
                "Email Text": [
                    "This is a long legitimate internal email about an engineering meeting.",
                    "Please verify your account password immediately at this fake login portal.",
                ],
                "Email Type": ["Safe Email", "Phishing Email"],
            }).to_csv(phishing_path, index=False)

            result = load_labeled_emails(enron_path, phishing_path)

        self.assertEqual(set(result["label"]), {"Ham", "Spam", "Phishing"})
        self.assertEqual(len(result), 3)

    def test_trained_pipeline_returns_class_probabilities(self):
        texts = [
            "team meeting agenda project schedule for the office",
            "normal work meeting and project status report",
            "buy cheap pills discount offer order now",
            "limited offer buy medicines at cheap prices",
            "verify account password immediately login security alert",
            "urgent account locked confirm password at fake login",
        ]
        labels = ["Ham", "Ham", "Spam", "Spam", "Phishing", "Phishing"]
        classifier = build_classifier().fit(texts, labels)
        request = EmailRequest(
            text="urgent verify account password and login",
            url_length=0,
            subdomain_count=0,
            https_presence=1,
            spf_dkim_failure=0,
            sender_reputation=0.5,
            recipient_count=1,
        )

        result = analyze_email(classifier, request)

        self.assertEqual(result["classification"], "Phishing")
        self.assertEqual(set(result["probabilities"]), {"Ham", "Spam", "Phishing"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(result["analysis_mode"], "trained_tfidf")
        self.assertTrue(result["xai_breakdown"])


if __name__ == "__main__":
    unittest.main()
