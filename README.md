# Big Five Personality Predictor

A machine learning web app that predicts your Big Five (OCEAN) personality traits from 25 questions, with SHAP-based explanations for each result.

Built with Streamlit, CatBoost, and SHAP.

🔗 **Live Demo:** [personalitytraitpredictor.streamlit.app](https://personalitytraitpredictor.streamlit.app/)

---

## What it does

- Presents 25 SHAP-ranked Likert-scale questions
- Runs your responses through 5 trained CatBoost models (one per trait)
- Shows your trait scores with a dominant trait highlight
- Explains predictions using SHAP waterfall plots
- Lets you download your full results as a CSV

## The Big Five Traits

| Trait | Description |
|---|---|
| Extraversion | Outgoing, energetic, socially driven |
| Neuroticism | Emotional reactivity and stress sensitivity |
| Agreeableness | Compassionate, cooperative, considerate |
| Conscientiousness | Organised, reliable, goal-directed |
| Openness | Curious, imaginative, open to new ideas |

---

## Project Structure

```
personality_trait_predictor/
├── app/
│   ├── main.py                  # Streamlit app
│   ├── model_utils.py           # Prediction pipeline
│   ├── adaptive_logic.py        # SHAP-ranked question selection
│   ├── cat_question_ranking.pkl # Pre-ranked question indices
│   └── static/
│       └── background.png
├── data/
│   ├── questions.py             # Question bank + trait names
│   ├── trait_descriptions.py    # Per-trait result descriptions
│   ├── X_train_filled_means.npy # Feature means for imputation
│   └── X_train_filled_df.pkl
├── model/
│   ├── final_catboost_models.pkl # 5 CatBoost models
│   ├── scaler.pkl                # StandardScaler for input features
│   └── imputer.pkl
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

## Requirements

- Python 3.9+
- See `requirements.txt` for full list

Key dependencies: `streamlit`, `catboost`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---

## How the prediction works

1. User answers 25 questions on a 1–5 Likert scale
2. Missing features are filled with per-feature training means
3. Inputs are standardized using the fitted `StandardScaler`
4. Each of the 5 CatBoost models produces a raw score
5. Scores are converted to percentages via temperature-scaled softmax
6. SHAP values explain which answers drove each trait score

---

## Privacy

No data is stored or transmitted. All processing happens locally in your browser session.
