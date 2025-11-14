from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"

CATEGORICAL = [
    "gender","Partner","PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
    "PaymentMethod"
]
NUMERIC = ["SeniorCitizen","DependentsCount","tenure","MonthlyCharges","TotalCharges"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # coerce TotalCharges to numeric (real datasets sometimes have blanks)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

def split(df: pd.DataFrame, test_size: float=0.2, seed: int=123) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df["Churn"], random_state=seed)

def feature_lists() -> Tuple[List[str], List[str]]:
    return NUMERIC, CATEGORICAL
