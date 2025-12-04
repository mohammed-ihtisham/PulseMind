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

## Setup

### 1. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training the Random Forest Model

First, train the core Random Forest model using the training script (make sure your virtual environment is activated):

```bash
python -m src.train
```

This will create the necessary model files in the `models/` directory.

## (Optional) Training the XGBoost Baseline

The research paper also reports results for a more complex **XGBoost** baseline model, trained on the same dataset and feature set. This baseline is **not used by the app**, but you can reproduce it for comparison.

### 1. Install extra dependency

Inside your virtual environment:

```bash
pip install xgboost
```

On macOS, XGBoost may also require the OpenMP runtime. If you see an error about `libomp.dylib` when importing xgboost, install it via Homebrew:

```bash
brew install libomp
```

### 2. Train the XGBoost model

From the project root, with the virtual environment activated:

```bash
python -m src.train_xgb
```

This will:

- Train an `XGBRegressor` with the hyperparameters described in the paper.
- Evaluate it on the same 80/20 train–test split used by the Random Forest and print MAE and R².
- Save the baseline model to `models/mental_health_xgb.pkl` for reference.

Again, the Streamlit app continues to use only the Random Forest model; the XGBoost baseline exists purely for experimental comparison.

## Running the Web Application

Launch the Streamlit app with (make sure your virtual environment is activated):

```bash
streamlit run app.py
```

The app will open in your default web browser, typically at `http://localhost:8501`.

### Features

- 🎨 **Modern UI/UX**: Beautiful gradient design with smooth animations
- 📊 **Interactive Visualizations**: Feature contribution charts and pie charts
- 🎯 **Real-time Predictions**: Get instant mental health score predictions
- 💡 **Personalized Recommendations**: AI-generated suggestions based on your habits
- 📱 **Responsive Design**: Works seamlessly on different screen sizes
