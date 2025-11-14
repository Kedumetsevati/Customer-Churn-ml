import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

MODEL_PATH = "models/churn_model.joblib"
META_PATH  = "models/model_meta.json"

# Load model
model = joblib.load(MODEL_PATH)

# Load best threshold if available
best_thr = 0.5
if Path(META_PATH).exists():
    with open(META_PATH) as f:
        try:
            best_thr = float(json.load(f).get("best_threshold", 0.5))
        except Exception:
            best_thr = 0.5

st.title("📉 Customer Churn Prediction")
st.caption("Telco-style demo — upload a CSV for batch scoring, try a single prediction, or view metrics.")

tab1, tab2, tab3 = st.tabs(["📦 Batch Scoring", "👤 Single Customer", "📊 Metrics & Monitoring"])

with tab1:
    st.subheader("Upload CSV to score (batch)")
    st.write("Columns should match training features (except `Churn`). Try `data/scoring_sample.csv`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        proba = model.predict_proba(df)[:,1]
        out = df.copy()
        out["churn_proba"] = proba
        out["churn_flag"]  = (out["churn_proba"] >= best_thr).astype(int)
        st.dataframe(out.head(20))
        st.download_button(
            "Download full predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Single-customer prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("gender", ["Male","Female"])
        senior = st.selectbox("SeniorCitizen", [0,1])
        depend = st.number_input("DependentsCount", min_value=0, max_value=10, value=0)
        phone = st.selectbox("PhoneService", ["Yes","No"])
        mult  = st.selectbox("MultipleLines", ["Yes","No"])
        internet = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
    with col2:
        online_sec = st.selectbox("OnlineSecurity", ["Yes","No"])
        online_bkp = st.selectbox("OnlineBackup", ["Yes","No"])
        device_prot = st.selectbox("DeviceProtection", ["Yes","No"])
        tech   = st.selectbox("TechSupport", ["Yes","No"])
        tv     = st.selectbox("StreamingTV", ["Yes","No"])
        movies = st.selectbox("StreamingMovies", ["Yes","No"])
    with col3:
        contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless = st.selectbox("PaperlessBilling", ["Yes","No"])
        pay       = st.selectbox("PaymentMethod", ["Electronic check","Mailed check","Bank transfer","Credit card"])
        tenure    = st.number_input("tenure (months)", min_value=0, max_value=120, value=6)
        monthly   = st.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
        total     = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=420.0, step=1.0)

    if st.button("Predict churn probability"):
        row = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":senior,"Partner":"No","DependentsCount":depend,
            "PhoneService":phone,"MultipleLines":mult,"InternetService":internet,
            "OnlineSecurity":online_sec,"OnlineBackup":online_bkp,"DeviceProtection":device_prot,
            "TechSupport":tech,"StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,
            "PaperlessBilling":paperless,"PaymentMethod":pay,"tenure":tenure,
            "MonthlyCharges":monthly,"TotalCharges":total
        }])
        proba = model.predict_proba(row)[:,1][0]
        st.metric("Churn probability", f"{proba:.2%}")
        st.caption("Tip: Month-to-month contracts, electronic checks, higher monthly charges, and short tenure often increase churn risk in this demo.")

with tab3:
    st.subheader("Test-set metrics & threshold tuning")
    if Path("data/test_scored.csv").exists():
        df = pd.read_csv("data/test_scored.csv")
        if "Churn" in df.columns and "churn_proba" in df.columns:
            y_true = (df["Churn"]=="Yes").astype(int).values
            st.write("Adjust decision threshold:")
            thr = st.slider("Threshold", 0.10, 0.90, float(best_thr), 0.01)
            y_pred = (df["churn_proba"] >= thr).astype(int).values

            # confusion matrix counts
            tn = int(((y_true==0)&(y_pred==0)).sum())
            fp = int(((y_true==0)&(y_pred==1)).sum())
            fn = int(((y_true==1)&(y_pred==0)).sum())
            tp = int(((y_true==1)&(y_pred==1)).sum())

            st.write("Confusion matrix (rows=True label; cols=Pred label)")
            st.dataframe(pd.DataFrame([[tn, fp],[fn, tp]], columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))

            # show top risky customers
            topn = df.sort_values("churn_proba", ascending=False).head(15)
            st.write("Top 15 high-risk customers")
            cols = [c for c in ["customerID","Contract","PaymentMethod","MonthlyCharges","tenure","churn_proba"] if c in df.columns]
            st.dataframe(topn[cols])
        else:
            st.info("Run training first to generate data/test_scored.csv with churn_proba.")
    else:
        st.info("No test-scored file found. Run: `python -m src.train`")
