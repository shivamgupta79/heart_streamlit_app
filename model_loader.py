# model_loader.py
import joblib
import json
import os
import numpy as np
import pandas as pd
import shap
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model_pipeline(filename="model_pipeline.joblib"):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Place pipeline joblib there.")
    pipeline = joblib.load(path)
    # pipeline should include preprocessing and final estimator (e.g., Pipeline([... , ('clf', XGBClassifier)]) )
    return pipeline

def get_model_metadata(filename="model_metadata.json"):
    p = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(p):
        return {"version": "unknown", "saved_at": None, "notes": ""}
    with open(p, "r") as f:
        return json.load(f)

def predict_df(pipeline, df):
    """
    df: pandas DataFrame of raw features (columns names expected)
    returns: DataFrame with `prob` and `pred` columns appended
    """
    # Pipeline expected to output probabilities via predict_proba
    X = df.copy()
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        # fallback: many scikit pipelines have last step as classifier; try to get proba from that
        try:
            probs = pipeline.named_steps['clf'].predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
        except Exception:
            preds = pipeline.predict(X)
            probs = preds.astype(float)
    out = df.copy()
    out["prob"] = probs
    out["pred"] = preds
    return out

def make_shap_explainer(pipeline, sample_X=None):
    """
    Create a SHAP explainer based on pipeline's final estimator.
    If the pipeline wraps the estimator, try to extract it.
    sample_X: small DataFrame used for background (if needed)
    """
    # try to grab estimator
    estimator = None
    # If pipeline is sklearn Pipeline, its last step is pipeline.steps[-1][1]
    try:
        from sklearn.pipeline import Pipeline as SKPipeline
        if isinstance(pipeline, SKPipeline):
            estimator = pipeline.steps[-1][1]
        else:
            estimator = pipeline
    except Exception:
        estimator = pipeline

    # If estimator is tree-based (XGBoost, RandomForest), use TreeExplainer
    try:
        if hasattr(estimator, "predict_proba") and ("XGB" in type(estimator).__name__.upper()
                                                   or "RANDOMFOREST" in type(estimator).__name__.upper()
                                                   or "LIGHTGBM" in type(estimator).__name__.upper()):
            explainer = shap.TreeExplainer(estimator)
            return explainer
    except Exception:
        pass

    # Fallback: KernelExplainer (slow). Use a small background sample if provided
    if sample_X is None:
        raise RuntimeError("Kernel explainer needs sample background. Provide sample_X DataFrame.")
    try:
        # create wrapper that applies pipeline preprocessing then estimator.predict_proba
        def predict_fn(x):
            # x is numpy array for background; convert to DataFrame with columns matching sample_X
            df = pd.DataFrame(x, columns=sample_X.columns)
            probs = pipeline.predict_proba(df)[:, 1]
            return probs
        explainer = shap.KernelExplainer(predict_fn, sample_X.iloc[:50].values)
        return explainer
    except Exception as e:
        raise RuntimeError("Could not create SHAP explainer: " + str(e))
