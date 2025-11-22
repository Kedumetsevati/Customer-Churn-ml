# streamlit_churn_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Customer Churn Prediction Dashboard – Telco Customers")
st.caption("Built by Kedumetse Nadour Vati | Random Forest + Streamlit")


st.markdown(
    """
This dashboard lets you explore **customer churn**, 
see **key patterns**, and view **high-risk customers** based on a machine learning model.

> 💡 Tip: Use the sidebar to upload your data and adjust the churn risk threshold.
"""
)

# --------------------------------------------------
# SIDEBAR: DATA INPUT
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.markdown("### 1. Upload or load data")

uploaded_file = st.sidebar.file_uploader(
    "Upload churn data (CSV)",
    type=["csv"],
    help="If you don’t upload anything, the app will try to load data/churn_data.csv"
)

target_col = st.sidebar.text_input(
    "Target column (churn flag)",
    value="Churn",
    help="Name of the column that has 0/1 or Yes/No for churn"
)

risk_threshold = st.sidebar.slider(
    "High-risk churn probability threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.7,
    step=0.01
)

@st.cache_data
def load_default_data():
    # Change this path to match your project
    try:
        df = pd.read_csv("data/churn_data.csv")
        return df
    except Exception as e:
        st.warning("No file uploaded and default file data/churn_data.csv not found.")
        return None

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

if df is None:
    st.stop()

st.success(f"Data loaded with **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------
# Try to convert Yes/No style churn labels to 0/1
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in data. Please check the name.")
    st.stop()

y_raw = df[target_col]

if y_raw.dtype == "O":
    # Object type – try to map Yes/No to 1/0
    y = y_raw.map({"Yes": 1, "No": 0, "Churn": 1, "No Churn": 0}).fillna(y_raw)
else:
    y = y_raw

# Force numeric 0/1 if possible
try:
    y = pd.to_numeric(y)
except Exception:
    st.error("Could not convert target column to numeric 0/1. Please clean the data.")
    st.stop()

X = df.drop(columns=[target_col])

# Identify numeric & categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

st.sidebar.markdown("### 2. Features used in the model")
st.sidebar.write("**Numeric:**", numeric_features if numeric_features else "None")
st.sidebar.write("**Categorical:**", categorical_features if categorical_features else "None")

if len(numeric_features) + len(categorical_features) == 0:
    st.error("No usable features found. Please check your dataset.")
    st.stop()

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
@st.cache_resource
def train_model(X, y, numeric_features, categorical_features):
    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    # Predictions
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)

    return clf, X_test, y_test, y_test_proba, auc, cm

with st.spinner("Training model..."):
    clf, X_test, y_test, y_test_proba, auc, cm = train_model(
        X, y, numeric_features, categorical_features
    )

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
st.markdown("## 🔑 Key Churn KPIs")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
total_churners = int(y.sum())
churn_rate = total_churners / total_customers * 100

# High-risk customers based on chosen threshold
full_proba = clf.predict_proba(X)[:, 1]
high_risk_mask = full_proba >= risk_threshold
high_risk_customers = int(high_risk_mask.sum())

avg_monthly_charge = None
for candidate in ["MonthlyCharges", "Monthly_Charges", "monthly_charges"]:
    if candidate in df.columns:
        avg_monthly_charge = df[candidate].mean()
        break

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Total Churners", f"{total_churners:,}", f"{churn_rate:.1f}% churn rate")

if avg_monthly_charge is not None:
    col3.metric("Avg Monthly Charge", f"${avg_monthly_charge:,.2f}")
else:
    col3.metric("Avg Monthly Charge", "N/A")

col4.metric(
    f"High-Risk Customers (p ≥ {risk_threshold:.2f})",
    f"{high_risk_customers:,}"
)

st.markdown("---")

# --------------------------------------------------
# LAYOUT: LEFT = OVERALL PATTERNS, RIGHT = MODEL
# --------------------------------------------------
left_col, right_col = st.columns([2, 1])

# ---------- LEFT: EXPLORATORY VISUALS ----------
with left_col:
    st.subheader("📉 Churn Distribution & Patterns")

    # Churn vs Non-Churn pie / bar
    churn_counts = y.value_counts().rename({0: "No Churn", 1: "Churn"}).reset_index()
    churn_counts.columns = ["ChurnStatus", "Count"]

    chart_churn = (
        alt.Chart(churn_counts)
        .mark_arc()
        .encode(
            theta="Count",
            color="ChurnStatus",
            tooltip=["ChurnStatus", "Count"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart_churn, use_container_width=True)

    # Churn rate by Contract if column exists
    if "Contract" in df.columns:
        st.markdown("#### Churn Rate by Contract Type")

        tmp = df.copy()
        tmp["ChurnFlag"] = y

        churn_by_contract = (
            tmp.groupby("Contract")["ChurnFlag"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnFlag": "ChurnRate"})
        )

        chart_contract = (
            alt.Chart(churn_by_contract)
            .mark_bar()
            .encode(
                x=alt.X("Contract:N", sort="-y"),
                y=alt.Y("ChurnRate:Q", axis=alt.Axis(format="%")),
                tooltip=["Contract", alt.Tooltip("ChurnRate:Q", format=".1%")]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_contract, use_container_width=True)

    # Churn rate by tenure group if 'tenure' exists
    if "tenure" in df.columns:
        st.markdown("#### Churn Rate by Tenure Group (months)")

        tmp = df.copy()
        tmp["ChurnFlag"] = y

        bins = [-np.inf, 12, 24, 48, np.inf]
        labels = ["0–12", "13–24", "25–48", "49+"]
        tmp["TenureGroup"] = pd.cut(tmp["tenure"], bins=bins, labels=labels)

        churn_by_tenure = (
            tmp.groupby("TenureGroup")["ChurnFlag"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnFlag": "ChurnRate"})
        )

        chart_tenure = (
            alt.Chart(churn_by_tenure)
            .mark_bar()
            .encode(
                x="TenureGroup:N",
                y=alt.Y("ChurnRate:Q", axis=alt.Axis(format="%")),
                tooltip=["TenureGroup", alt.Tooltip("ChurnRate:Q", format=".1%")]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_tenure, use_container_width=True)

# ---------- RIGHT: MODEL PERFORMANCE ----------
with right_col:
    st.subheader("🤖 Model Performance")

    st.metric("ROC AUC", f"{auc:.3f}")

    st.markdown("**Confusion Matrix (threshold = 0.50)**")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    )

    cm_df_long = cm_df.reset_index().melt("index")
    cm_df_long.columns = ["Actual", "Predicted", "Count"]

    cm_chart = (
        alt.Chart(cm_df_long)
        .mark_rect()
        .encode(
            x="Predicted:N",
            y="Actual:N",
            color="Count:Q",
            tooltip=["Actual", "Predicted", "Count"]
        )
        .properties(height=250)
    )

    st.altair_chart(cm_chart, use_container_width=True)

    # Feature importance (from RF model)
    st.markdown("**Top Feature Importances**")

    rf_model = clf.named_steps["model"]
    preprocessor = clf.named_steps["preprocessor"]

    # Build feature names after preprocessing
    feature_names = []

    # Numeric
    feature_names.extend(numeric_features)

    # Categorical
    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)

    importances = rf_model.feature_importances_

    fi_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fi_chart = (
        alt.Chart(fi_df)
        .mark_bar()
        .encode(
            x="importance:Q",
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature", "importance"]
        )
        .properties(height=400)
    )

    st.altair_chart(fi_chart, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# HIGH-RISK CUSTOMERS TABLE
# --------------------------------------------------
st.subheader("🚨 High-Risk Customers")

df_with_proba = df.copy()
df_with_proba["churn_proba"] = full_proba

high_risk_df = df_with_proba[df_with_proba["churn_proba"] >= risk_threshold].copy()
high_risk_df_sorted = high_risk_df.sort_values("churn_proba", ascending=False)

st.markdown(
    f"Showing customers with **churn probability ≥ {risk_threshold:.2f}** "
    f"({len(high_risk_df_sorted)} rows)."
)

max_rows_to_show = st.slider(
    "Number of rows to display",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

st.dataframe(
    high_risk_df_sorted.head(max_rows_to_show),
    use_container_width=True
)

st.download_button(
    label="💾 Download high-risk customers as CSV",
    data=high_risk_df_sorted.to_csv(index=False).encode("utf-8"),
    file_name="high_risk_customers.csv",
    mime="text/csv"
)
