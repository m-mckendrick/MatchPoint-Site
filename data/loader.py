import joblib
import pandas as pd
from pathlib import Path
from src.feature_engineering.feature_builder import build_features

#change based on what model you wish to use
MODEL_PATH = Path("models/logreg_baseline.pkl")
MATCHES_PATH = Path("data/processed/matches")
RANKINGS_PATH = Path("data/processed/atp_rankings_2000-2020.csv")

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("scaler", None)

def load_matches():
    #load all processed match CSVs into a single DataFrame
    files = list(Path(MATCHES_PATH).glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    return df


def load_matches_features():
    df = load_matches()
    #apply feature builder (training mode)
    df = build_features(df)
    #make sure is in datetime
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    return df


def load_rankings():
    df = pd.read_csv(RANKINGS_PATH)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    return df
