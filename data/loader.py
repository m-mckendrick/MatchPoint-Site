import joblib
import pandas as pd
from pathlib import Path

#change based on what model you wish to use
RAW_MATCHES_PATH = Path("data/raw")
MATCHES_PATH = Path("data/processed/matches")
PROCESSED_PATH = Path("data/processed")
FEATURES_PATH = Path(PROCESSED_PATH / f"2000-2023_features.csv")
RANKINGS_PATH = Path("data/processed/atp_rankings_2000-2020.csv")
SHAP_PATH = Path("models/shap_explainer_xgboost_3.pkl")

def load_model(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("scaler", None), bundle.get("features", None)

def load_model_shap():
    shap = joblib.load(SHAP_PATH)
    return shap

def load_raw_matches():
    files = list(Path(MATCHES_PATH).glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    return df

def load_matches():
    #load all processed match CSVs into a single DataFrame
    files = list(Path(MATCHES_PATH).glob("*_clean.csv"))

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    return df


def load_rankings():
    df = pd.read_csv(RANKINGS_PATH)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    return df

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    return df