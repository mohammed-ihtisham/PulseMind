# 🧠 PulseMind - Transparent AI Tool for Digital Well-Being

This repo trains an interpretable model that predicts a composite mental health score
(from a digital habits dataset) and explains which factors contributed most to that prediction.

## Project Structure

- `data/digital_habits_vs_mental_health.csv` — Kaggle dataset
- `models/` — saved model and explainer artifacts
- `src/utils.py` — shared helpers (paths, risk categorization)
- `src/train.py` — trains the RandomForest model and SHAP explainer
- `src/predict.py` — loads artifacts and runs predictions with explanations
- `app.py` — Streamlit web application with modern UI/UX

## Setup (all commands from repo root)

1) Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

2) Install core dependencies

```bash
pip install -r requirements.txt
```

## Train models

### Train the primary Random Forest (used by the app)

```bash
source venv/bin/activate
python -m src.train
```

Outputs:
- `models/mental_health_model.pkl`
- `models/feature_names.json`

### (Optional) Train the XGBoost baseline

```bash
source venv/bin/activate
pip install xgboost               # macOS may also need: brew install libomp
python -m src.train_xgb
```

Outputs:
- `models/mental_health_xgb.pkl` (baseline only; app still uses the Random Forest)

## Run predictions from the command line

Interactive demo (prompts for inputs and prints results):

```bash
source venv/bin/activate
python -m src.predict
```

Programmatic example:

```bash
source venv/bin/activate
python - <<'PY'
from src.predict import predict_mental_health
features = {
    "screen_time_hours": 6.5,
    "social_media_platforms_used": 3,
    "hours_on_TikTok": 1.5,
    "sleep_hours": 7.0,
}
print(predict_mental_health(features))
PY
```

## Run the Streamlit app

```bash
source venv/bin/activate
streamlit run app.py
```

The app opens at `http://localhost:8501`. Ensure the Random Forest model artifacts exist in `models/` (run `python -m src.train` if not).

### Features

- 🎨 **Modern UI/UX**: Beautiful gradient design with smooth animations
- 📊 **Interactive Visualizations**: Feature contribution charts and pie charts
- 🎯 **Real-time Predictions**: Get instant mental health score predictions
- 💡 **Personalized Recommendations**: AI-generated suggestions based on your habits
- 📱 **Responsive Design**: Works seamlessly on different screen sizes
