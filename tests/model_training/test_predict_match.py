import pandas as pd
from pathlib import Path
from src.model_training.logistical_regression import prepare_dataset
from src.feature_engineering.feature_builder import build_features
from src.model_training.predict_match import load_model, build_match

DATA_PATH = Path("data/processed")

def test_prediction_pipeline():
    # load model and scaler
    model, scaler = load_model()

    # load data + feature list
    df = build_features(DATA_PATH)
    x, y = prepare_dataset(df)
    model_features = x.columns.tolist()

    # pick two known players in dataset
    p1 = "Novak Djokovic"
    p2 = "Rafael Nadal"

    # pick a cutoff date
    cutoff_date = pd.to_datetime("2019-01-01")

    # build match row
    row = build_match(df, p1, p2, model_features, cutoff_date)

    # check shape
    assert isinstance(row, pd.DataFrame)
    assert list(row.columns) == model_features
    assert row.shape == (1, len(model_features))

    # scale
    row_scaled = scaler.transform(row)

    # predict
    prob = model.predict_proba(row_scaled)[0][1]

    # check probability is valid
    assert 0 <= prob <= 1
