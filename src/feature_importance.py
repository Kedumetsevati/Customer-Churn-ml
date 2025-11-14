import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from src.utils import load_data, split, feature_lists, TARGET

DATA_PATH = "data/sample_churn.csv"
MODEL_PATH = "models/churn_model.joblib"

def main():
    df = load_data(DATA_PATH)
    train_df, test_df = split(df)
    y_test = (test_df[TARGET]=="Yes").astype(int)
    num_cols, cat_cols = feature_lists()
    X_test = test_df[num_cols + cat_cols]

    model = joblib.load(MODEL_PATH)
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    r = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=123, scoring="roc_auc"
    )
    imp = pd.DataFrame({
        "feature": num_cols + cat_cols,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    Path("reports").mkdir(exist_ok=True)
    imp.to_csv("reports/permutation_importance.csv", index=False)
    print(f"Base AUC: {base_auc:.4f}")
    print("Top drivers:\n", imp.head(10))
    print("Saved reports/permutation_importance.csv")

if __name__ == "__main__":
    main()
