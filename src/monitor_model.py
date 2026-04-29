from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
NEW_BATCH_PATH = PROJECT_ROOT / "data" / "processed" / "new_scoring_batch.csv"

def main():
    baseline_df = pd.read_csv(BASELINE_PATH)
    baseline_df["TotalCharges"] = pd.to_numeric(
        baseline_df["TotalCharges"], errors="coerce"
    )

    if not NEW_BATCH_PATH.exists():
        print("No new scoring batch found.")
        return

    new_df = pd.read_csv(NEW_BATCH_PATH)
    new_df["TotalCharges"] = pd.to_numeric(
        new_df.get("TotalCharges"), errors="coerce"
    )

    print("Baseline shape:", baseline_df.shape)
    print("New batch shape:", new_df.shape)
    print("\\nMissing values in new batch:")
    print(new_df.isnull().mean().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
