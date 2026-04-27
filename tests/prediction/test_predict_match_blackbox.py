import pandas as pd
import joblib
from data.loader import load_model ,load_rankings, load_features
from src.prediction.match_builder_modular import build_match
from src.prediction.predictor import predict
from src.model_training.logistical_regression import prepare_dataset

#testing inputs to output
def test_predict_match_blackbox():
    # load model and data
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    df = load_features()
    ranking_df = load_rankings()


    # build a synthetic match
    row_model, row_full = build_match(
        df=df,
        rankings_df = ranking_df,
        p1_name="Novak Djokovic",
        p2_name="Daniil Medvedev",
        tournament="US Open",
        year=2018,
        round_label="F",
        model_features=model_features,
        debug=False
    )

    # ensure row is a DataFrame with correct columns
    assert isinstance(row_model, pd.DataFrame)
    assert list(row_model.columns) == model_features

    #ensure full row correct
    assert isinstance(row_full, pd.DataFrame)
    assert row_full.shape[0] == 1
    assert "p1_name" in row_full.columns
    assert "p2_name" in row_full.columns
    assert "tourney_name" in row_full.columns

    # Scale + predict
    prob = predict(model, scaler, row_model)

    # Probability must be between 0 and 1
    assert 0 <= prob <= 1