# Transparent AI Tool for Digital Well-Being — Core Model

This repo trains an interpretable model that predicts a composite mental health score
(from a digital habits dataset) and explains which factors contributed most to that prediction.

## Project Structure

- `data/digital_habits_vs_mental_health.csv` — Kaggle dataset
- `models/` — saved model and explainer artifacts
- `src/utils.py` — shared helpers (paths, risk categorization)
- `src/train.py` — trains the RandomForest model and SHAP explainer
- `src/predict.py` — loads artifacts and runs predictions with explanations

## Setup

```bash
pip install -r requirements.txt
