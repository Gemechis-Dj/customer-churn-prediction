from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importance.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_metrics.json"

def main():
    if not FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError("Run src/train_model.py first.")

    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    print("Top 10 Important Features")
    print("-" * 40)
    print(importance_df.head(10).to_string(index=False))

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\\nBest Model:", metrics.get("model_name"))
        print("ROC-AUC:", metrics.get("roc_auc"))

if __name__ == "__main__":
    main()
