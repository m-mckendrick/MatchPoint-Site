import joblib
import pandas as pd
from pathlib import Path

#change based on what model you wish to use
MODEL_PATH = Path("models/logreg_baseline.pkl")
RAW_MATCHES_PATH = Path("data/raw")
MATCHES_PATH = Path("data/processed/matches")
PROCESSED_PATH = Path("data/processed")
RANKINGS_PATH = Path("data/processed/atp_rankings_2000-2020.csv")

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("scaler", None)

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
