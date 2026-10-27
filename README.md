# Local Hybrid Email Defender

The project includes a Chrome extension for analyzing the currently open Gmail
message with the FastAPI service running on your own computer. The extension
does not require a deployed server.

## Run locally

### Train the classifier once

From a terminal in the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m app.training.train_email_classifier
```

Alternatively, double-click `train_model.bat`.

If your virtual environment is named `venv`, use
`.\venv\Scripts\python.exe` instead. The first run downloads the public datasets
(about 90 MB), removes duplicate emails, creates stratified train/validation/test
splits, trains a CPU-friendly text classifier, and saves the model and measured
metrics under `app\models\artifacts\`. Run this command again when you want to
retrain. Use `--max-per-class 1000` for a quicker smoke run.

### Start the API and extension

1. Start the API from the project folder with `start.bat`. Keep its console
   window open. The API listens on `127.0.0.1:8000`.
2. In Chrome, open `chrome://extensions` and enable **Developer mode**.
3. Select **Load unpacked** and choose this project's `extension` folder.
4. Open a Gmail message. Click **Analyze open email** in the lower-right corner
   and review the result.

The extension only runs on Gmail and only sends a message for analysis after
you click its button. It contacts only the local API at
`http://127.0.0.1:8000/analyze`. Stop the API with Ctrl+C in the server window.

## Datasets and model scope

Training combines two public datasets:

- [Enron Spam Email Dataset](https://huggingface.co/datasets/adamlouly/enron_spam_data),
  declared Apache-2.0. Its labels map `0` to Ham and `1` to Spam.
- [Phishing Email Dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset),
  declared LGPL-3.0. Its Safe Email and Phishing Email labels map to Ham and
  Phishing.

The model is a TF-IDF word/character n-gram classifier trained on email text.
The generated report records source URLs, class counts, held-out accuracy,
macro-F1, and a confusion matrix. The existing benchmark charts are illustrative
and are not measurements of this classifier. The held-out metrics do not
guarantee performance on current or future mail; treat predictions as advisory
and verify suspicious messages independently.

The original Hybrid ML-DNN components remain in the repository but are not used
for this training path. The current classifier does not use sender metadata,
SPF/DKIM results, or URL reputation.

Gmail's message view does not expose SPF/DKIM validation or sender reputation
to this extension. It sends the visible email text to the local API; metadata
values supplied by the extension are not used by this text-only classifier.
