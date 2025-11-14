import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.utils import load_data, split, feature_lists, TARGET

DATA_PATH = "data/sample_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/model_meta.json"

def pick_best_threshold(y_true, proba, metric="f1"):
    # search thresholds 0.10 .. 0.90
    thresholds = np.linspace(0.1, 0.9, 33)
    scores = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, pred, pos_label=1)
        else:
            # Youden's J for ROC operating point
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            s = tpr - fpr
        scores.append((t, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0], scores

def main():
    df = load_data(DATA_PATH)
    train_df, test_df = split(df)

    y_train = (train_df[TARGET]=="Yes").astype(int)
    y_test  = (test_df[TARGET]=="Yes").astype(int)

    num_cols, cat_cols = feature_lists()
    X_train = train_df[num_cols + cat_cols]
    X_test  = test_df[num_cols + cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = ImbPipeline(steps=[
        ("prep", pre),
        ("smote", SMOTE(random_state=123)),
        ("logit", LogisticRegression(max_iter=200))
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    best_t, curve = pick_best_threshold(y_test, proba, metric="f1")
    preds_05 = (proba >= 0.5).astype(int)
    preds_bt = (proba >= best_t).astype(int)

    print(f"AUC: {auc:.4f}")
    print("\n== 0.50 threshold ==")
    print(confusion_matrix(y_test, preds_05))
    print(classification_report(y_test, preds_05, digits=3))
    print(f"\n== Best F1 threshold: {best_t:.2f} ==")
    print(confusion_matrix(y_test, preds_bt))
    print(classification_report(y_test, preds_bt, digits=3))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    meta = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "auc": float(auc),
        "best_threshold": float(best_t)
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # Save scored test set with probabilities (for dashboards)
    out_test = test_df.copy()
    out_test["churn_proba"] = proba
    out_test.to_csv("data/test_scored.csv", index=False)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata (incl. best_threshold={best_t:.2f}) to {META_PATH}")
    print("Wrote data/test_scored.csv")

if __name__ == "__main__":
    main()
