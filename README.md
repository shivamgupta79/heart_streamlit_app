# Heart Risk Predictor — Streamlit Prototype

A lightweight Streamlit app that lets doctors or patients upload chest/ECG/csv features (or enter them manually) to get:

* a risk **probability** and binary **prediction** for heart disease,
* a **recommendation** (e.g., refer to cardiologist if prob ≥ threshold),
* **SHAP**-based top-5 feature explanation for each patient,
* **model versioning** and audit **logging** to a simple SQLite DB.

This repository contains a ready-to-run prototype intended for demos and internal evaluation. **Not for clinical use** without full validation and regulatory approval.

---

## File structure

```
heart_streamlit_app/
│
├─ app.py                    # main Streamlit app
├─ model_loader.py           # model load/predict/SHAP helpers
├─ db.py                     # sqlite helpers for logs & versions
├─ models/
│   ├─ model_pipeline.joblib  # your preprocessing + classifier pipeline (joblib)
│   └─ model_metadata.json    # JSON metadata (version, date, description)
│
├─ requirements.txt
└─ README.md                 # this file
```

---

## Features

* Upload CSV (rows = patients) or manual entry form
* Prediction probability + predicted class + recommendation
* SHAP per-patient top-5 feature bar chart
* Logs each prediction to `predictions.db` (SQLite)
* Basic admin panel to view recent logs (password protected)
* Model hot-swap via models/ folder or sidebar upload

---

## Prerequisites

* Python 3.9+ recommended
* `models/model_pipeline.joblib` — a scikit-learn `Pipeline` that encapsulates preprocessing **and** the final estimator, and supports `predict_proba`.

  * The pipeline must accept a pandas DataFrame (with expected column names) as input and output probabilities via `predict_proba`.
* Optional: `models/model_metadata.json` with metadata (see example below)

---

## Example model_metadata.json

```json
{
  "version": "v1.0",
  "saved_at": "2025-11-29T12:00:00Z",
  "notes": "XGBoost tuned with SMOTE",
  "expected_columns": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
}
```

---

## Installation

1. Clone the repo

```bash
git clone https://github.com/youruser/heart_streamlit_app.git
cd heart_streamlit_app
```

2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
shap
matplotlib
seaborn
python-dotenv
sqlalchemy
imblearn
```

(If you trained a Keras model and want to load it, add `tensorflow`.)

---

## Prepare and save your trained model

Inside your training notebook, create a single `Pipeline` that includes preprocessing (imputation, encoding, scaling or ColumnTransformer) and the classifier as the last step:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# Example: column transformers and final estimator (replace with your own)
# preproc = ColumnTransformer([...])
# clf = best_estimator (e.g., tuned XGBoost)
pipeline = Pipeline([
    ('preproc', preproc),
    ('clf', best_estimator)
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/model_pipeline.joblib")

# save metadata
import json
meta = {"version":"v1.0", "saved_at":"2025-11-29T12:00:00Z", "notes":"tuned xgboost + smote", "expected_columns": list(X_train.columns)}
with open("models/model_metadata.json","w") as f:
    json.dump(meta, f)
```

**Important:** `pipeline.predict_proba(df)` must work where `df` is a pandas DataFrame with column names matching `expected_columns`.

---

## Running the app (local)

```bash
streamlit run app.py
```

* The app will create `predictions.db` (SQLite) automatically in the project folder.
* Use the sidebar to set the referral threshold (default 0.6).
* Admin tab requires a password — set `STREAMLIT_ADMIN_PASS` environment variable to override the default. Example:

```bash
export STREAMLIT_ADMIN_PASS="supersecret"
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push the repository to GitHub including `app.py`, `model_loader.py`, `db.py`, `requirements.txt`, and `models/` (or omit large model and upload at runtime via the app sidebar).
2. Create a Streamlit Cloud account and connect your repository.
3. Configure environment variable `STREAMLIT_ADMIN_PASS` via the Streamlit Cloud settings.
4. Deploy.

---

## Deployment alternatives

* **Render / Heroku:** Can deploy Streamlit apps using Docker or buildpacks (Streamlit Cloud is simplest).
* **Docker:** Create Dockerfile to run `streamlit run app.py` and deploy to any container host.
* For enterprise/hospital deployment prefer a backend (FastAPI/Flask) with proper auth, HTTPS, and hosting behind hospital firewall.

---

## Security & Privacy

This app stores input JSONs in `predictions.db`. For production:

* **Do not** store PII or store only hashed identifiers.
* Use HTTPS, network-level protections, and role-based authentication.
* Encrypt the DB at rest or store logs in a secure server with audit controls.
* Comply with local privacy/regulatory guidance (e.g., India’s data protection rules, hospital policies).

---

## Validation & Monitoring

* Validate the model on external datasets before clinical use.
* Track metrics over time (AUC, recall, false negatives) and implement drift detection.
* Log the ground-truth outcomes (when available) to re-evaluate and retrain.

---

## Troubleshooting

* **`Model file not found`**: Put `model_pipeline.joblib` inside `models/` directory.
* **`predict_proba` errors**: Ensure the saved pipeline supports `predict_proba` and accepts DataFrames.
* **SHAP slow / errors**: For non-tree models, KernelExplainer is used and can be slow; provide a representative background sample to speed up or precompute explanations for common patterns.

---

## Extending the prototype

* Add PDF export (report with prediction + SHAP chart) using `reportlab` or `weasyprint`.
* Add user authentication (OAuth, hospital SSO).
* Add a more polished front-end (React/Next.js) and convert Streamlit to ML backend endpoints (FastAPI).
* Add monitoring dashboards and automatic retraining pipelines.

---

## License & Disclaimer

* **License:** MIT (or specify your preferred license).
* **Disclaimer:** This prototype is for educational/demo purposes only. It is **not** a medical device and must not be used as a standalone diagnostic tool. Clinicians should verify and validate predictions before acting.

---

## Contact / Maintainer

* Maintainer: [Your Name] (replace in repo)
* Repo: `https://github.com/youruser/heart_streamlit_app` (replace with your repo)
* For help preparing `model_pipeline.joblib` from your notebook, include your training notebook or paste `expected_columns` and I’ll provide the exact code to wrap preprocessing + classifier into a pipeline.

---

Thank you — paste this `README.md` into the project root. If you want, I can now:

* generate a ready-to-commit `README.md` file for you (I can paste it as a file content),
* OR produce a single ZIP/GitHub-ready repo tree including `app.py`, `model_loader.py`, `db.py`, and `requirements.txt`.
