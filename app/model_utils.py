# app/model_utils.py

import os
import pickle
import pandas as pd
import numpy as np

<<<<<<< HEAD
=======
# Base directory of the current file
>>>>>>> e0f835fb9a30035ecb276be692e27fb2ef34f6da
BASE_DIR = os.path.dirname(__file__)

def load_model():
    path = os.path.join(BASE_DIR, "../model/final_catboost_models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_means():
<<<<<<< HEAD
    """Returns the per-feature imputation means (numpy array, shape 50)."""
    path = os.path.join(BASE_DIR, "../data/X_train_filled_means.npy")
    return np.load(path)
=======
    path = os.path.join(BASE_DIR, "../data/X_train_filled_df.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
>>>>>>> e0f835fb9a30035ecb276be692e27fb2ef34f6da

def load_scaler():
    path = os.path.join(BASE_DIR, "../model/scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

<<<<<<< HEAD
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
    temperature = 0.15  # lower = more separation between traits
    exp_p = np.exp((preds - preds.max()) / temperature)
    return (exp_p / exp_p.sum()) * 100
=======
def load_imputer():
    path = os.path.join(BASE_DIR, "../model/imputer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_traits(models_dict, input_vector):
    preds = []

    # Convert to DataFrame if input is numpy array
    if isinstance(input_vector, np.ndarray):
        feature_names = [f"Q{i+1}" for i in range(50)]
        input_df = pd.DataFrame(input_vector.reshape(1, -1), columns=feature_names)
    else:
        input_df = input_vector  # Already a DataFrame

    for trait, model in models_dict.items():
        pred = model.predict(input_df)
        preds.append(pred[0] if isinstance(pred, (list, np.ndarray)) else pred)

    return np.array(preds)

def inverse_standardize(preds):
    """
    Reverses standardization back to 1–5 Likert scale.
    Original labels were standardized (mean=0, std=1) from [1, 5].
    """
    return np.clip((preds * 0.75) + 3, 1, 5)

def score_to_percentage(score):
    """
    Converts a Likert score (1–5) to a percentage [0% - 100%].
    """
    return ((score - 1) / 4) * 100
>>>>>>> e0f835fb9a30035ecb276be692e27fb2ef34f6da
