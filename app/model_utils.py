# app/model_utils.py

import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)

def load_model():
    path = os.path.join(BASE_DIR, "../model/final_catboost_models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_means():
    """Returns the per-feature imputation means (numpy array, shape 50)."""
    path = os.path.join(BASE_DIR, "../data/X_train_filled_means.npy")
    return np.load(path)

def load_scaler():
    path = os.path.join(BASE_DIR, "../model/scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_traits(models_dict, raw_responses, scaler):
    """
    Predict all 5 traits.
    raw_responses: numpy array of shape (50,) with Likert 1-5 values (NaN filled).
    scaler: fitted StandardScaler used during training.
    Returns raw model outputs (standardized label space).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = scaler.transform(raw_responses.reshape(1, -1))

    input_df = pd.DataFrame(scaled, columns=[str(i) for i in range(50)])

    preds = []
    for trait, model in models_dict.items():
        pred = model.predict(input_df)
        preds.append(float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred))

    return np.array(preds)

def scores_to_softmax_pct(raw_preds):
    """
    Converts raw model outputs to relative percentages using temperature-scaled softmax.
    Lower temperature = more contrast between traits.
    """
    preds = np.array(raw_preds, dtype=float)
    temperature = 0.15
    exp_p = np.exp((preds - preds.max()) / temperature)
    return (exp_p / exp_p.sum()) * 100
