from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_prediction_pipeline.pkl"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importance.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_metrics.json"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def ensure_directories():
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "metrics").mkdir(parents=True, exist_ok=True)

def load_data():
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        df.to_csv(RAW_DATA_PATH, index=False)
    return df

def clean_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    churn_map = {
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
        "true": 1, "false": 0
    }

    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(churn_map)
    )

    df = df.dropna(subset=["Churn"]).copy()

    if df.empty:
        raise ValueError("No valid rows found after cleaning target column 'Churn'.")

    df["Churn"] = df["Churn"].astype(int)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

def extract_feature_importance(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = abs(model.coef_[0])
    else:
        importance_values = [0.0] * len(feature_names)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    }).sort_values("importance", ascending=False)

    return importance_df

def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }
    return metrics

def main():
    ensure_directories()
    df = clean_data(load_data())

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),
    }

    best_name = None
    best_pipeline = None
    best_metrics = None
    best_auc = -1.0

    for name, model in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)

        print(f"\\n{name}")
        print("-" * 60)
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    joblib.dump(best_pipeline, MODEL_PATH)
    extract_feature_importance(best_pipeline).to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print("\\nBest model saved:", best_name)
    print("Model path:", MODEL_PATH)
    print("Feature importance path:", FEATURE_IMPORTANCE_PATH)
    print("Metrics path:", METRICS_PATH)

if __name__ == "__main__":
    main()
