from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_prediction_pipeline.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result = df.copy()
    result["churn_prediction"] = predictions
    result["churn_probability"] = probabilities
    result["churn_label"] = result["churn_prediction"].map({
        1: "Likely to Churn",
        0: "Likely to Stay"
    })
    return result
