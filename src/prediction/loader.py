import joblib
import pandas as pd
from pathlib import Path
from src.feature_engineering.feature_builder import build_features

MODEL_PATH = Path("models/logreg_baseline.pkl")
MATCHES_PATH = Path("data/processed/matches")
RANKINGS_PATH = Path("data/processed/atp_rankings_2000-2020.csv")

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("scaler", None)

def load_matches():
    return build_features(MATCHES_PATH)

def load_rankings():
    df = pd.read_csv(RANKINGS_PATH)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    return df
