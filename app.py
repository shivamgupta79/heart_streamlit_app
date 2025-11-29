# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from model_loader import load_model_pipeline, get_model_metadata, predict_df, make_shap_explainer
import db
import os

st.set_page_config(layout="wide", page_title="Heart Risk Predictor (Demo)")

# Initialize DB
db.init_db()

# Sidebar: load model
st.sidebar.title("Model & Settings")
models_dir = os.path.join(os.path.dirname(__file__), "models")
model_file = st.sidebar.file_uploader("(Optional) Upload new model_pipeline.joblib", type=["joblib", "pkl"])
# If user uploaded a model file via the sidebar, save to models dir
if model_file is not None:
    with open(os.path.join(models_dir, "model_pipeline.joblib"), "wb") as f:
        f.write(model_file.getbuffer())
    st.sidebar.success("Saved model_pipeline.joblib to models/ - restart app to load it.")

# Load pipeline from disk
try:
    pipeline = load_model_pipeline()
    metadata = get_model_metadata()
    model_version = metadata.get("version", "v?") if isinstance(metadata, dict) else "v?"
    st.sidebar.success(f"Loaded model (version: {model_version})")
except Exception as e:
    st.sidebar.error("Could not load model pipeline. Put model_pipeline.joblib in models/ folder.")
    st.stop()

# slider threshold
THRESH = st.sidebar.slider("Referral threshold (probability)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

# Admin password (very basic): set in environment var STREAMLIT_ADMIN_PASS or default 'admin'
ADMIN_PASS = os.getenv("STREAMLIT_ADMIN_PASS", "admin123")

# Tabs: Predict / Admin
tab = st.sidebar.radio("Mode", ["Predict", "Admin"])

if tab == "Predict":
    st.title("Heart Disease Risk Predictor — Demo")
    st.markdown(
        """
        Upload a CSV or use the manual form. The app will run preprocessing (the same pipeline used in training),
        return risk probability, predicted class, and show top-5 SHAP contributors for a selected patient.
        """
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Input")
        upload = st.file_uploader("Upload CSV (rows = patients)", type=["csv"])
        if upload is not None:
            df_input = pd.read_csv(upload)
            st.write("Preview input:")
            st.dataframe(df_input.head())
        else:
            st.info("Or fill in the manual patient form below.")
            # Manual form: we try to detect numeric features expected by your pipeline.
            # Build a minimal form using pipeline expected input columns if present
            # Attempt to infer columns from pipeline (if pipeline is a sklearn pipeline and keeps column names)
            try:
                # Try to get expected feature names from a training example saved as metadata
                sample_cols = metadata.get("expected_columns", None) if isinstance(metadata, dict) else None
            except Exception:
                sample_cols = None

            if sample_cols is None:
                # Fallback: present a small set of typical Cleveland features
                sample_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

            manual = {}
            with st.form("manual_form"):
                st.write("Manual patient entry (fill fields):")
                for c in sample_cols:
                    # Try numeric input
                    try:
                        val = st.text_input(c, value="")
                        manual[c] = val
                    except Exception:
                        manual[c] = ""
                submitted = st.form_submit_button("Create single-row dataset")
            if submitted:
                # convert to DataFrame (attempt numeric cast)
                df_input = pd.DataFrame([manual])
                # coerce numeric where possible
                for col in df_input.columns:
                    df_input[col] = pd.to_numeric(df_input[col], errors='ignore')
                st.write("Input row:")
                st.dataframe(df_input)
            else:
                df_input = None

    with col2:
        st.header("Predictions")
        if 'df_input' in locals() and df_input is not None:
            try:
                preds_df = predict_df(pipeline, df_input)
                st.write("Predictions (showing probability & predicted class):")
                st.dataframe(preds_df[["prob", "pred"]].head())
                # Add recommendation
                preds_df["recommendation"] = preds_df["prob"].apply(lambda p: "Refer to cardiologist" if p >= THRESH else "Monitor / Routine check")
                st.table(preds_df[["prob","pred","recommendation"]])
                # Log predictions
                for idx, row in preds_df.iterrows():
                    db.log_prediction(model_version, row.drop(["prob","pred","recommendation"]).to_dict(), float(row["prob"]), int(row["pred"]), user="web", notes="")
                st.success("Predictions done and logged.")
            except Exception as e:
                st.error("Prediction error: " + str(e))
        else:
            st.info("Upload CSV or fill manual form to get predictions.")

    st.markdown("---")
    st.header("Explainability (SHAP)")

    if 'preds_df' in locals():
        st.write("Select row to explain:")
        row_idx = st.number_input("Row index", min_value=0, max_value=max(0, len(preds_df)-1), value=0)
        row = df_input.iloc[[row_idx]]
        try:
            # Build a small background for explainer from the uploaded CSV if available, else try to use pipeline metadata sample
            background = None
            if upload is not None:
                # use sample of df_input as background
                background = df_input.sample(min(50, len(df_input)))
            else:
                # attempt to create background from single row by repeating it (not ideal)
                background = row
            explainer = make_shap_explainer(pipeline, sample_X=background if background is not None else None)
            # For tree explainer we can pass the transformed features if pipeline transforms. But to be robust,
            # we call explainer on the preprocessed data if estimator expects transformed features
            # We'll try to obtain X to pass to explainer by running pipeline's preprocessing if present.
            # If pipeline is sklearn Pipeline, we attempt to call pipeline[:-1].transform to get features.
            X_for_shap = None
            try:
                from sklearn.pipeline import Pipeline as SKPipeline
                if isinstance(pipeline, SKPipeline):
                    # if pipeline has named steps and has preprocessors, extract all except final estimator
                    if len(pipeline.steps) > 1:
                        preprocess = SKPipeline(pipeline.steps[:-1])
                        X_for_shap = preprocess.transform(row)
                    else:
                        X_for_shap = row.values
                else:
                    X_for_shap = row.values
            except Exception:
                X_for_shap = row.values

            # compute shap values (vector) for the selected row
            shap_vals = explainer(X_for_shap)  # object
            # shap_vals may be an Explanation object; get array-like values
            try:
                vals = np.array(shap_vals.values).reshape(-1)
                feature_names = shap_vals.feature_names if hasattr(shap_vals, "feature_names") else None
            except Exception:
                # fallback: try to compute using pipeline.predict_proba via KernelExplainer output
                vals = np.array(shap_vals[0]).reshape(-1)
                feature_names = background.columns if background is not None else None

            # Create a dataframe of contributions
            if feature_names is None:
                # try from df_input columns
                feature_names = (background.columns if background is not None else row.columns)

            contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": vals})
            contrib_df["abs_val"] = contrib_df["shap_value"].abs()
            topk = contrib_df.sort_values("abs_val", ascending=False).head(5).sort_values("shap_value")
            # Plot horizontal bar
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x="shap_value", y="feature", data=topk, ax=ax)
            ax.set_title("Top 5 SHAP feature contributions (this patient)")
            st.pyplot(fig)
            # Show table of contributions
            st.table(topk.drop(columns=["abs_val"]).assign(shap_value=lambda df: df.shap_value.round(4)))
        except Exception as e:
            st.error("Could not compute SHAP: " + str(e))

elif tab == "Admin":
    st.title("Admin / Logs")
    pwd = st.text_input("Admin password", type="password")
    if pwd != ADMIN_PASS:
        st.warning("Enter admin password to view logs.")
    else:
        st.success("Authenticated")
        # show last 200 records
        preds = db.fetch_predictions(limit=200)
        if len(preds) == 0:
            st.info("No predictions logged yet.")
        else:
            # Flatten input JSON for display
            rows = []
            for p in preds:
                row = {
                    "id": p["id"],
                    "ts": p["ts"],
                    "model_version": p["model_version"],
                    "prob": p["prob"],
                    "pred": p["pred"],
                    "user": p["user"],
                    "notes": p["notes"]
                }
                # flatten some input keys (avoid too many columns)
                try:
                    inp = p["input"]
                    # include age and sex if exist to give quick filter columns
                    row["age"] = inp.get("age", "")
                    row["sex"] = inp.get("sex", "")
                except Exception:
                    pass
                rows.append(row)
            df_logs = pd.DataFrame(rows)
            st.dataframe(df_logs)
            # simple filters
            high_risk = df_logs[df_logs["prob"] >= THRESH]
            st.write(f"High risk (prob >= {THRESH}): {len(high_risk)}")
            st.dataframe(high_risk.head(50))

st.markdown("---")
st.caption("This prototype is for demo/educational use only. Not a clinical device.")
