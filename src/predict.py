import sys
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = "models/churn_model.joblib"

def predict(input_csv: str, output_csv: str="data/predictions.csv"):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv)
    proba = model.predict_proba(df)[:,1]
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_flag"] = (out["churn_proba"] >= 0.5).astype(int)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(out)} rows")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py data/scoring_sample.csv [output_csv]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/predictions.csv"
    predict(inp, out)
